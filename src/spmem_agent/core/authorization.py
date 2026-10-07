from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import threading
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple


_PRIVACY_TYPE_ALIASES = {
    "person_name": "name",
    "phone": "phone_number",
    "insurance_record": "insurance",
}


def normalize_privacy_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _PRIVACY_TYPE_ALIASES.get(text, text)


@dataclass(frozen=True)
class PrivacyReleaseAuthorization:
    """Opaque, request-scoped capability for releasing exact private values."""

    token: str
    session_id: str
    user_id: str
    allowed_privacy_types: Tuple[str, ...]
    expires_at: datetime


class PrivacyReleaseAuthorizer:
    """Issues short-lived capabilities after the protected-store confirmation."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = ttl_seconds
        self._active: Dict[str, PrivacyReleaseAuthorization] = {}
        self._lock = threading.RLock()

    def issue(
        self,
        *,
        session_id: str,
        user_id: str,
        allowed_privacy_types: Iterable[str],
    ) -> PrivacyReleaseAuthorization:
        allowed = tuple(
            dict.fromkeys(
                normalized
                for value in allowed_privacy_types
                if (normalized := normalize_privacy_type(value))
            )
        )
        if not allowed:
            raise ValueError("at least one privacy type must be authorized")

        grant = PrivacyReleaseAuthorization(
            token=uuid.uuid4().hex,
            session_id=str(session_id),
            user_id=str(user_id),
            allowed_privacy_types=allowed,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds),
        )
        with self._lock:
            self._active[grant.token] = grant
        return grant

    def validate(
        self,
        authorization: Optional[PrivacyReleaseAuthorization],
        *,
        user_id: str,
        privacy_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        if not isinstance(authorization, PrivacyReleaseAuthorization):
            return False

        with self._lock:
            active = self._active.get(authorization.token)

        if active != authorization:
            return False
        if datetime.now(timezone.utc) >= active.expires_at:
            self.revoke(active)
            return False
        if str(active.user_id) != str(user_id):
            return False
        if session_id is not None and str(active.session_id) != str(session_id):
            return False

        normalized_type = normalize_privacy_type(privacy_type)
        if normalized_type and normalized_type not in active.allowed_privacy_types:
            return False
        return True

    def revoke(self, authorization: Optional[PrivacyReleaseAuthorization]) -> None:
        if not isinstance(authorization, PrivacyReleaseAuthorization):
            return
        with self._lock:
            self._active.pop(authorization.token, None)

