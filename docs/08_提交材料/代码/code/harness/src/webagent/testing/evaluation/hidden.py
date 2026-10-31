"""Read-only access to the site's hidden state for external judging.

The judge opens a new SQLite connection, sets ``query_only`` and reads only
hidden truth, activity, drafts and submission counts. It does not import the
site's page handlers and it cannot change site data.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence


class HiddenStateError(RuntimeError):
    """Hidden database is missing or not initialized."""


class HiddenStateReader:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise HiddenStateError(f"hidden state database not found: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _rows(self, sql: str, values: Sequence[Any] = ()) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]
        finally:
            connection.close()

    # ------------------------------------------------------------------- data
    def meta(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self._rows("SELECT key, value FROM meta")}

    def products(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM products ORDER BY position")

    def posts(self) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM posts ORDER BY position")
        for row in rows:
            row["tags"] = json.loads(row["tags"])
        return rows

    def hidden_truth(
        self, *, role: str | None = None, item_type: str | None = None
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if role is not None:
            clauses.append("role = ?")
            values.append(role)
        if item_type is not None:
            clauses.append("item_type = ?")
            values.append(item_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return self._rows(
            f"SELECT * FROM hidden_truth{where} ORDER BY item_type, item_id", values
        )

    # --------------------------------------------------------------- activity
    def activities(
        self,
        *,
        kinds: Sequence[str] | None = None,
        item_type: str | None = None,
        item_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if kinds:
            clauses.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            values.extend(kinds)
        if item_type is not None:
            clauses.append("item_type = ?")
            values.append(item_type)
        if item_id is not None:
            clauses.append("item_id = ?")
            values.append(item_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._rows(f"SELECT * FROM activity{where} ORDER BY seq", values)
        for row in rows:
            row["payload"] = json.loads(row["payload"]) if row["payload"] else {}
        return rows

    def viewed_ids(self, item_type: str) -> set[str]:
        return {
            str(row["item_id"])
            for row in self.activities(kinds=["view_item"], item_type=item_type)
            if row["item_id"] is not None
        }

    def draft_texts(self, item_type: str, item_id: str) -> list[str]:
        texts: list[str] = []
        for row in self.activities(
            kinds=["draft_saved", "draft_cancelled"],
            item_type=item_type,
            item_id=item_id,
        ):
            text = row["payload"].get("text")
            if isinstance(text, str):
                texts.append(text)
        return texts

    def saved_draft_texts(self, item_type: str, item_id: str) -> list[str]:
        """Texts recorded by Save draft actions, oldest first."""

        texts: list[str] = []
        for row in self.activities(
            kinds=["draft_saved"], item_type=item_type, item_id=item_id
        ):
            text = row["payload"].get("text")
            if isinstance(text, str):
                texts.append(text)
        return texts

    def current_draft(self, item_type: str, item_id: str) -> str | None:
        """The persisted field value, or None when the field has no saved value."""

        rows = self._rows(
            "SELECT text FROM drafts WHERE item_type = ? AND item_id = ?",
            (item_type, item_id),
        )
        return None if not rows else str(rows[0]["text"])

    def cancelled(self, item_type: str, item_id: str) -> bool:
        return bool(
            self.activities(
                kinds=["draft_cancelled"], item_type=item_type, item_id=item_id
            )
        )

    def submission_count(self, item_id: str | None = None) -> int:
        clauses = ["kind = 'comment_sent'"]
        values: list[Any] = []
        if item_id is not None:
            clauses.append("item_id = ?")
            values.append(item_id)
        rows = self._rows(
            "SELECT COUNT(*) AS count FROM activity WHERE " + " AND ".join(clauses),
            values,
        )
        return int(rows[0]["count"])

    def fault_events(self) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM fault_events ORDER BY seq")
        for row in rows:
            row["truth"] = json.loads(row["truth"])
        return rows

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, Any]:
        meta = self.meta()
        products = self._rows("SELECT * FROM products ORDER BY id")
        posts = self.posts()
        for row in posts:
            row["tags"] = list(row["tags"])
        posts.sort(key=lambda row: row["id"])
        truth = self._rows("SELECT * FROM hidden_truth ORDER BY item_type, item_id")
        faults = self._rows("SELECT * FROM faults ORDER BY fault_id")
        for row in faults:
            row["enabled"] = bool(row["enabled"])
            row["params"] = json.loads(row["params"]) if row["params"] else {}
        config_rows = self._rows(
            "SELECT value FROM hidden_config WHERE key = 'config'"
        )
        hidden_config = (
            json.loads(config_rows[0]["value"]) if config_rows else {}
        )
        return {
            "meta": meta,
            "products": products,
            "posts": posts,
            "hidden_truth": truth,
            "faults": faults,
            "hidden_config": hidden_config,
        }

    def snapshot_hash(self) -> str:
        payload = json.dumps(
            self.snapshot(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = ["HiddenStateError", "HiddenStateReader"]
