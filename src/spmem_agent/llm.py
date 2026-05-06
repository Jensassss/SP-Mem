from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from openai import OpenAI


def build_openai_llm_call(
    *,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Callable[[str, str], str]:
    """
    Build llm_call(system_prompt, user_prompt) for PrivacyAwareAgent.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    extra_params = extra_params or {}

    def llm_call(system_prompt: str, user_prompt: str) -> str:
        params: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(extra_params)

        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        return content or ""

    return llm_call

