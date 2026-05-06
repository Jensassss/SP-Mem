import logging
import sqlite3
import threading
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLiteManager:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._migrate_history_table()
        self._create_history_table()
        self._create_privacy_tables()
        # Backfill legacy rows (privacy_mappings.memory_id) into link table once.
        self._backfill_privacy_mapping_links()
        self.connection.commit()

    def _create_privacy_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS privacy_mappings (
                id TEXT PRIMARY KEY,
                memory_id TEXT,
                user_id TEXT NOT NULL,
                privacy_type TEXT NOT NULL,
                raw_value TEXT NOT NULL,
                raw_hash TEXT NOT NULL,
                sanitized_value TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )

        self.connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_privacy_mapping_unique
            ON privacy_mappings(user_id, privacy_type, raw_hash)
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS privacy_mapping_links (
                mapping_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (mapping_id, memory_id)
            )
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_privacy_mapping_links_memory
            ON privacy_mapping_links(memory_id)
            """
        )

    def _backfill_privacy_mapping_links(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO privacy_mapping_links (mapping_id, memory_id, created_at, updated_at)
            SELECT id, memory_id, COALESCE(created_at, ?), COALESCE(updated_at, ?)
            FROM privacy_mappings
            WHERE memory_id IS NOT NULL
            """,
            (now, now),
        )

    def _migrate_history_table(self) -> None:
        """
        If a pre-existing history table had the old group-chat columns,
        rename it, create the new schema, copy the intersecting data, then
        drop the old table.
        """
        with self._lock:
            try:
                # Start a transaction
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()

                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                if cur.fetchone() is None:
                    self.connection.execute("COMMIT")
                    return  # nothing to migrate

                cur.execute("PRAGMA table_info(history)")
                old_cols = {row[1] for row in cur.fetchall()}

                expected_cols = {
                    "id",
                    "memory_id",
                    "old_memory",
                    "new_memory",
                    "event",
                    "created_at",
                    "updated_at",
                    "is_deleted",
                    "actor_id",
                    "role",
                }

                if old_cols == expected_cols:
                    self.connection.execute("COMMIT")
                    return

                logger.info("Migrating history table to new schema (no convo columns).")

                # Clean up any existing history_old table from previous failed migration
                cur.execute("DROP TABLE IF EXISTS history_old")

                # Rename the current history table
                cur.execute("ALTER TABLE history RENAME TO history_old")

                # Create the new history table with updated schema
                cur.execute(
                    """
                    CREATE TABLE history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                """
                )

                # Copy data from old table to new table
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cur.execute(f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old")

                # Drop the old table
                cur.execute("DROP TABLE history_old")

                # Commit the transaction
                self.connection.execute("COMMIT")
                logger.info("History table migration completed successfully.")

            except Exception as e:
                # Rollback the transaction on any error
                self.connection.execute("ROLLBACK")
                logger.error(f"History table migration failed: {e}")
                raise

    def _create_history_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                """
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create history table: {e}")
                raise

    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add history record: {e}")
                raise

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role
                FROM history
                WHERE memory_id = ?
                ORDER BY created_at ASC, DATETIME(updated_at) ASC
            """,
                (memory_id,),
            )
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "old_memory": r[2],
                "new_memory": r[3],
                "event": r[4],
                "created_at": r[5],
                "updated_at": r[6],
                "is_deleted": bool(r[7]),
                "actor_id": r[8],
                "role": r[9],
            }
            for r in rows
        ]

    def reset(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS history")
                self.connection.execute("DROP TABLE IF EXISTS privacy_mappings")
                self.connection.execute("DROP TABLE IF EXISTS privacy_mapping_links")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                """
                )
                self._create_privacy_tables()
                self.connection.commit()
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to reset tables: {e}")
                raise

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()

    def get_privacy_mapping(self, user_id: str, privacy_type: str, raw_hash: str):
        with self._lock:
            cursor = self.connection.execute(
                """
                SELECT id, memory_id, user_id, privacy_type, raw_value, raw_hash,
                    sanitized_value, created_at, updated_at
                FROM privacy_mappings
                WHERE user_id = ? AND privacy_type = ? AND raw_hash = ?
                """,
                (user_id, privacy_type, raw_hash),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "memory_id": row[1],
            "user_id": row[2],
            "privacy_type": row[3],
            "raw_value": row[4],
            "raw_hash": row[5],
            "sanitized_value": row[6],
            "created_at": row[7],
            "updated_at": row[8],
        }
    
    def upsert_privacy_mapping(
        self,
        user_id: str,
        privacy_type: str,
        raw_value: str,
        sanitized_value: str,
        memory_id: str | None = None,
    ):
        raw_hash = hashlib.md5(str(raw_value).encode()).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        mapping_id = str(uuid.uuid4())

        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO privacy_mappings (
                        id, memory_id, user_id, privacy_type, raw_value, raw_hash,
                        sanitized_value, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, privacy_type, raw_hash)
                    DO UPDATE SET
                        memory_id = CASE
                            WHEN excluded.memory_id IS NULL THEN privacy_mappings.memory_id
                            ELSE excluded.memory_id
                        END,
                        sanitized_value = excluded.sanitized_value,
                        updated_at = excluded.updated_at
                    """,
                    (
                        mapping_id,
                        memory_id,
                        user_id,
                        privacy_type,
                        raw_value,
                        raw_hash,
                        sanitized_value,
                        now,
                        now,
                    ),
                )

                # Resolve the stable mapping id after upsert.
                cursor = self.connection.execute(
                    """
                    SELECT id FROM privacy_mappings
                    WHERE user_id = ? AND privacy_type = ? AND raw_hash = ?
                    """,
                    (user_id, privacy_type, raw_hash),
                )
                row = cursor.fetchone()
                resolved_mapping_id = row[0] if row else None

                if resolved_mapping_id and memory_id:
                    self.connection.execute(
                        """
                        INSERT INTO privacy_mapping_links (mapping_id, memory_id, created_at, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(mapping_id, memory_id)
                        DO UPDATE SET updated_at = excluded.updated_at
                        """,
                        (resolved_mapping_id, memory_id, now, now),
                    )

                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise

        return self.get_privacy_mapping(user_id, privacy_type, raw_hash)
    
    def link_mapping_to_memory(self, mapping_id: str, memory_id: str):
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    UPDATE privacy_mappings
                    SET memory_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (memory_id, now, mapping_id),
                )
                self.connection.execute(
                    """
                    INSERT INTO privacy_mapping_links (mapping_id, memory_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(mapping_id, memory_id)
                    DO UPDATE SET updated_at = excluded.updated_at
                    """,
                    (mapping_id, memory_id, now, now),
                )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise

    def list_privacy_mappings_by_memory(self, memory_id: str):
        with self._lock:
            cursor = self.connection.execute(
                """
                SELECT pm.id, pml.memory_id, pm.user_id, pm.privacy_type, pm.raw_value, pm.raw_hash,
                    pm.sanitized_value, pm.created_at, pm.updated_at
                FROM privacy_mapping_links pml
                JOIN privacy_mappings pm
                  ON pm.id = pml.mapping_id
                WHERE pml.memory_id = ?
                """,
                (memory_id,),
            )
            rows = cursor.fetchall()

            if not rows:
                # Backward compatibility for legacy data before link table.
                cursor = self.connection.execute(
                    """
                    SELECT id, memory_id, user_id, privacy_type, raw_value, raw_hash,
                        sanitized_value, created_at, updated_at
                    FROM privacy_mappings
                    WHERE memory_id = ?
                    """,
                    (memory_id,),
                )
                rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "memory_id": row[1],
                "user_id": row[2],
                "privacy_type": row[3],
                "raw_value": row[4],
                "raw_hash": row[5],
                "sanitized_value": row[6],
                "created_at": row[7],
                "updated_at": row[8],
            }
            for row in rows
        ]
    
    def delete_privacy_mappings_by_memory(self, memory_id: str):
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                # Bring legacy rows into link table first, then unlink current memory.
                self._backfill_privacy_mapping_links()
                self.connection.execute(
                    """
                    DELETE FROM privacy_mapping_links
                    WHERE memory_id = ?
                    """,
                    (memory_id,),
                )
                self.connection.execute(
                    """
                    UPDATE privacy_mappings
                    SET memory_id = NULL, updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (now, memory_id),
                )

                # Remove only orphan mappings (no memory links remain).
                self.connection.execute(
                    """
                    DELETE FROM privacy_mappings
                    WHERE id IN (
                        SELECT pm.id
                        FROM privacy_mappings pm
                        LEFT JOIN privacy_mapping_links pml
                          ON pml.mapping_id = pm.id
                        WHERE pml.mapping_id IS NULL
                    )
                    """
                )

                # Refresh representative memory_id for surviving mappings.
                self.connection.execute(
                    """
                    UPDATE privacy_mappings
                    SET memory_id = (
                            SELECT pml.memory_id
                            FROM privacy_mapping_links pml
                            WHERE pml.mapping_id = privacy_mappings.id
                            ORDER BY pml.updated_at DESC
                            LIMIT 1
                        ),
                        updated_at = ?
                    WHERE id IN (
                        SELECT DISTINCT mapping_id
                        FROM privacy_mapping_links
                    )
                    """,
                    (now,),
                )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        
