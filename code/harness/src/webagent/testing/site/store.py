"""SQLite storage for the local dynamic website.

Every HTTP request opens its own short-lived connection, so several browser
tabs observe the same drafts, activities and submission counts. Hidden truth,
fault definitions and fault-trigger records live in separate tables that no
page handler reads or renders.

The store is test-side code. It is not registered as an Agent tool and has no
HTTP route.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data import Dataset

_SCHEMA = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE products (
    id TEXT PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    rating REAL NOT NULL,
    stock INTEGER NOT NULL,
    summary TEXT NOT NULL,
    description TEXT NOT NULL,
    position INTEGER NOT NULL
);
CREATE TABLE posts (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    body TEXT NOT NULL,
    likes INTEGER NOT NULL,
    comments INTEGER NOT NULL,
    tags TEXT NOT NULL,
    created_label TEXT NOT NULL,
    position INTEGER NOT NULL
);
CREATE TABLE hidden_truth (
    item_type TEXT NOT NULL,
    item_id TEXT NOT NULL,
    role TEXT NOT NULL,
    relevance TEXT,
    tone TEXT,
    expected_group TEXT,
    PRIMARY KEY (item_type, item_id)
);
CREATE INDEX hidden_truth_role ON hidden_truth (role);
CREATE TABLE hidden_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE drafts (
    item_type TEXT NOT NULL,
    item_id TEXT NOT NULL,
    text TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    saved INTEGER NOT NULL,
    PRIMARY KEY (item_type, item_id)
);
CREATE TABLE activity (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    item_type TEXT,
    item_id TEXT,
    path TEXT,
    payload TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX activity_kind ON activity (kind);
CREATE INDEX activity_item ON activity (item_type, item_id);
CREATE TABLE faults (
    fault_id TEXT PRIMARY KEY,
    fault_type TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    trigger_event TEXT NOT NULL,
    trigger_after INTEGER NOT NULL,
    params TEXT NOT NULL,
    note TEXT
);
CREATE TABLE fault_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    fault_id TEXT NOT NULL,
    fault_type TEXT NOT NULL,
    trigger_event TEXT NOT NULL,
    observed_count INTEGER NOT NULL,
    truth TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE site_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SiteStore:
    """Repository for one isolated case/run database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    # ------------------------------------------------------------- lifecycle
    def _drop_schema(self) -> None:
        tables = (
            "meta",
            "products",
            "posts",
            "hidden_truth",
            "hidden_config",
            "drafts",
            "activity",
            "faults",
            "fault_events",
            "site_state",
        )
        with self._connect() as connection:
            for table in tables:
                connection.execute(f"DROP TABLE IF EXISTS {table}")

    def initialize(
        self,
        dataset: Dataset,
        *,
        case_id: str,
        case_version: str,
        hidden_config: Mapping[str, Any],
        faults: Iterable[Mapping[str, Any]],
        page_size: int,
        setup: Mapping[str, Any],
    ) -> None:
        self._drop_schema()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [
                    ("case_id", case_id),
                    ("case_version", case_version),
                    ("seed", str(dataset.seed)),
                    ("page_size", str(page_size)),
                    ("generator_params", json.dumps(dataset.generator_params, ensure_ascii=False)),
                    ("setup", json.dumps(setup, ensure_ascii=False)),
                ],
            )
            connection.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        product.id,
                        product.sku,
                        product.name,
                        product.category,
                        product.price_cents,
                        product.rating,
                        product.stock,
                        product.summary,
                        product.description,
                        product.position,
                    )
                    for product in dataset.products
                ],
            )
            connection.executemany(
                "INSERT INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        post.id,
                        post.title,
                        post.author,
                        post.body,
                        post.likes,
                        post.comments,
                        json.dumps(list(post.tags), ensure_ascii=False),
                        post.created_label,
                        post.position,
                    )
                    for post in dataset.posts
                ],
            )
            connection.executemany(
                "INSERT INTO hidden_truth VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        record.item_type,
                        record.item_id,
                        record.role,
                        record.relevance,
                        record.tone,
                        record.expected_group,
                    )
                    for record in dataset.hidden
                ],
            )
            connection.execute(
                "INSERT INTO hidden_config(key, value) VALUES ('config', ?)",
                (json.dumps(hidden_config, ensure_ascii=False),),
            )
            for fault in faults:
                connection.execute(
                    "INSERT OR REPLACE INTO faults VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(fault["fault_id"]),
                        str(fault["type"]),
                        int(bool(fault.get("enabled", False))),
                        str(fault.get("trigger_event", "")),
                        int(fault.get("trigger_after", 0)),
                        json.dumps(fault.get("params", {}), ensure_ascii=False),
                        fault.get("note"),
                    ),
                )

    def reset(
        self,
        dataset: Dataset,
        *,
        case_id: str,
        case_version: str,
        hidden_config: Mapping[str, Any],
        faults: Iterable[Mapping[str, Any]],
        page_size: int,
        setup: Mapping[str, Any],
    ) -> None:
        # All request connections close before reset is called. Rebuilding the
        # schema instead of deleting the file keeps Windows file locks simple.
        self.initialize(
            dataset,
            case_id=case_id,
            case_version=case_version,
            hidden_config=hidden_config,
            faults=faults,
            page_size=page_size,
            setup=setup,
        )

    # ------------------------------------------------------------------ meta
    def meta(self) -> dict[str, str]:
        with self._connect() as connection:
            return {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM meta")
            }

    def hidden_config(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM hidden_config WHERE key = 'config'"
            ).fetchone()
        return json.loads(row["value"]) if row else {}

    # -------------------------------------------------------------- products
    def list_products(
        self,
        *,
        query: str = "",
        category: str = "",
        min_price_cents: int | None = None,
        max_price_cents: int | None = None,
        min_rating: float | None = None,
        in_stock: bool = False,
        page: int = 1,
        page_size: int = 8,
    ) -> tuple[list[sqlite3.Row], int]:
        clauses: list[str] = []
        values: list[Any] = []
        if query:
            clauses.append("(name LIKE ? OR summary LIKE ? OR category LIKE ?)")
            pattern = f"%{query}%"
            values.extend([pattern, pattern, pattern])
        if category:
            clauses.append("category = ?")
            values.append(category)
        if min_price_cents is not None:
            clauses.append("price_cents >= ?")
            values.append(int(min_price_cents))
        if max_price_cents is not None:
            clauses.append("price_cents <= ?")
            values.append(int(max_price_cents))
        if min_rating is not None:
            clauses.append("rating >= ?")
            values.append(float(min_rating))
        if in_stock:
            clauses.append("stock > 0")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        offset = max(0, (page - 1) * page_size)
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) AS count FROM products{where}", values
            ).fetchone()["count"]
            rows = connection.execute(
                f"SELECT * FROM products{where} ORDER BY position LIMIT ? OFFSET ?",
                [*values, page_size, offset],
            ).fetchall()
        return list(rows), int(total)

    def all_products(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(connection.execute("SELECT * FROM products ORDER BY position"))

    def get_product(self, product_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM products WHERE id = ?", (product_id,)
            ).fetchone()

    def categories(self) -> list[str]:
        with self._connect() as connection:
            return [
                row["category"]
                for row in connection.execute(
                    "SELECT DISTINCT category FROM products ORDER BY category"
                )
            ]

    # ----------------------------------------------------------------- posts
    def list_posts(
        self, *, query: str = "", page: int = 1, page_size: int = 8
    ) -> tuple[list[sqlite3.Row], int]:
        offset = max(0, (page - 1) * page_size)
        where = ""
        values: list[Any] = []
        if query:
            where = " WHERE (title LIKE ? OR body LIKE ? OR tags LIKE ?)"
            pattern = f"%{query}%"
            values = [pattern, pattern, pattern]
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) AS count FROM posts{where}", values
            ).fetchone()["count"]
            rows = connection.execute(
                f"SELECT * FROM posts{where} ORDER BY position LIMIT ? OFFSET ?",
                [*values, page_size, offset],
            ).fetchall()
        return list(rows), int(total)

    def get_post(self, post_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM posts WHERE id = ?", (post_id,)
            ).fetchone()

    # ---------------------------------------------------------------- drafts
    def get_draft(self, item_type: str, item_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT text FROM drafts WHERE item_type = ? AND item_id = ?",
                (item_type, item_id),
            ).fetchone()
        return "" if row is None else str(row["text"])

    def save_draft(self, item_type: str, item_id: str, text: str) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO drafts(item_type, item_id, text, updated_at, saved) "
                "VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(item_type, item_id) DO UPDATE SET "
                "text = excluded.text, updated_at = excluded.updated_at, saved = 1",
                (item_type, item_id, text, now),
            )
            self._record_activity(
                connection,
                "draft_saved",
                item_type=item_type,
                item_id=item_id,
                payload={"text": text, "length": len(text)},
            )

    def cancel_draft(self, item_type: str, item_id: str, text: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM drafts WHERE item_type = ? AND item_id = ?",
                (item_type, item_id),
            )
            self._record_activity(
                connection,
                "draft_cancelled",
                item_type=item_type,
                item_id=item_id,
                payload={"text": text, "length": len(text)},
            )

    def send_comment(self, item_type: str, item_id: str, text: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM drafts WHERE item_type = ? AND item_id = ?",
                (item_type, item_id),
            )
            self._record_activity(
                connection,
                "comment_sent",
                item_type=item_type,
                item_id=item_id,
                payload={"text": text, "length": len(text)},
            )

    # -------------------------------------------------------------- activity
    @staticmethod
    def _record_activity(
        connection: sqlite3.Connection,
        kind: str,
        *,
        item_type: str | None = None,
        item_id: str | None = None,
        path: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO activity(kind, item_type, item_id, path, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                kind,
                item_type,
                item_id,
                path,
                json.dumps(payload or {}, ensure_ascii=False),
                utc_now(),
            ),
        )

    def record_activity(
        self,
        kind: str,
        *,
        item_type: str | None = None,
        item_id: str | None = None,
        path: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            self._record_activity(
                connection,
                kind,
                item_type=item_type,
                item_id=item_id,
                path=path,
                payload=payload,
            )

    def activity_count(self, kind: str, *, item_id: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS count FROM activity WHERE kind = ?"
        values: list[Any] = [kind]
        if item_id is not None:
            sql += " AND item_id = ?"
            values.append(item_id)
        with self._connect() as connection:
            return int(connection.execute(sql, values).fetchone()["count"])

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
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM activity{where} ORDER BY seq", values
            ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload"]) if row["payload"] else {}
            result.append(
                {
                    "seq": int(row["seq"]),
                    "kind": row["kind"],
                    "item_type": row["item_type"],
                    "item_id": row["item_id"],
                    "path": row["path"],
                    "payload": payload,
                    "created_at": row["created_at"],
                }
            )
        return result

    def submission_count(self, item_id: str | None = None) -> int:
        return self.activity_count("comment_sent", item_id=item_id)

    # ----------------------------------------------------------- hidden truth
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
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM hidden_truth{where} ORDER BY item_type, item_id", values
            ).fetchall()
        return [
            {
                "item_type": row["item_type"],
                "item_id": row["item_id"],
                "role": row["role"],
                "relevance": row["relevance"],
                "tone": row["tone"],
                "expected_group": row["expected_group"],
            }
            for row in rows
        ]

    def hidden_snapshot(self) -> dict[str, Any]:
        """Stable data/truth/fault fingerprint; run activity is excluded."""

        with self._connect() as connection:
            meta = {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM meta")
            }
            products = [
                dict(row)
                for row in connection.execute("SELECT * FROM products ORDER BY id")
            ]
            posts = [
                dict(row)
                for row in connection.execute("SELECT * FROM posts ORDER BY id")
            ]
            truth = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM hidden_truth ORDER BY item_type, item_id"
                )
            ]
            faults = [
                dict(row)
                for row in connection.execute("SELECT * FROM faults ORDER BY fault_id")
            ]
        for row in posts:
            row["tags"] = json.loads(row["tags"])
        for row in faults:
            row["enabled"] = bool(row["enabled"])
            row["params"] = json.loads(row["params"]) if row["params"] else {}
        config_row = self.hidden_config()
        return {
            "meta": meta,
            "products": products,
            "posts": posts,
            "hidden_truth": truth,
            "faults": faults,
            "hidden_config": config_row,
        }

    # ------------------------------------------------------------------ faults
    def faults(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM faults ORDER BY fault_id").fetchall()
        return [
            {
                "fault_id": row["fault_id"],
                "type": row["fault_type"],
                "enabled": bool(row["enabled"]),
                "trigger_event": row["trigger_event"],
                "trigger_after": int(row["trigger_after"]),
                "params": json.loads(row["params"]) if row["params"] else {},
                "note": row["note"],
            }
            for row in rows
        ]

    def add_fault(self, fault: Mapping[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO faults VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(fault["fault_id"]),
                    str(fault["type"]),
                    int(bool(fault.get("enabled", True))),
                    str(fault.get("trigger_event", "")),
                    int(fault.get("trigger_after", 0)),
                    json.dumps(fault.get("params", {}), ensure_ascii=False),
                    fault.get("note"),
                ),
            )

    def fault_events(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM fault_events ORDER BY seq"
            ).fetchall()
        return [
            {
                "seq": row["seq"],
                "fault_id": row["fault_id"],
                "fault_type": row["fault_type"],
                "trigger_event": row["trigger_event"],
                "observed_count": row["observed_count"],
                "truth": json.loads(row["truth"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def mark_fault_fired(self, fault_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE faults SET enabled = 0 WHERE fault_id = ?", (fault_id,)
            )

    def record_fault_event(
        self,
        *,
        fault_id: str,
        fault_type: str,
        trigger_event: str,
        observed_count: int,
        truth: Mapping[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO fault_events(fault_id, fault_type, trigger_event, "
                "observed_count, truth, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fault_id,
                    fault_type,
                    trigger_event,
                    observed_count,
                    json.dumps(truth, ensure_ascii=False),
                    utc_now(),
                ),
            )

    # ------------------------------------------------------------- site state
    def state_get(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM site_state WHERE key = ?", (key,)
            ).fetchone()
        return default if row is None else str(row["value"])

    def state_set(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO site_state(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def state_delete(self, key: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM site_state WHERE key = ?", (key,))


__all__ = ["SiteStore", "utc_now"]
