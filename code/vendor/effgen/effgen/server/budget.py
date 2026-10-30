"""Per-principal daily cost tracking for RBAC cost caps.

Each authenticated principal accrues spend against the ``max_cost_per_day``
of its effective policy. When the accrued spend for the current UTC day meets
or exceeds the cap, further charged requests are rejected with
:class:`BudgetExceeded` (surfaced as HTTP 429 by the API server).

The tracker is process-local and in-memory by default. A daily snapshot is
optionally persisted to ``~/.effgen/budget/<YYYY-MM-DD>.json`` so that spend
survives a restart within the same day; set ``EFFGEN_BUDGET_PERSIST=0`` to
disable persistence.

This is intentionally simple — a single-process counter. For multi-replica
deployments back it with a shared store (Redis); the public API
(:func:`charge`, :func:`get_spend`, :func:`reset`) is the integration point.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from effgen.errors import quote_for_message, with_next_step

logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """Raised when a principal exceeds its daily cost cap.

    Surfaced by the API server as HTTP 429 (Too Many Requests). The message
    names the cap that was reached and then what clears it.

    Attributes:
        status_code: The HTTP status this maps to (429 by default).
    """

    _FOLLOW_ON = (
        "The cap resets at the start of the next UTC day. Raise the "
        "principal's daily limit in the budget configuration, or route the "
        "request to a cheaper model."
    )

    def __init__(self, reason: str, status_code: int = 429) -> None:
        super().__init__(with_next_step(quote_for_message(reason), self._FOLLOW_ON))
        self.status_code = status_code


# ---------------------------------------------------------------------------
# In-memory daily ledger
# ---------------------------------------------------------------------------

_lock = threading.Lock()
# {day -> {principal -> spend_usd}}
_ledger: dict[str, dict[str, float]] = {}
# In-flight reservations: {day -> {principal -> {reservation_id -> amount}}}.
# Reserved (but not yet committed) amounts count against the cap so concurrent
# requests cannot all slip past the check before any of them is charged.
_reservations: dict[str, dict[str, dict[str, float]]] = {}
_BUDGET_DIR: Path | None = None


def _today() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


def _persist_enabled() -> bool:
    return os.getenv("EFFGEN_BUDGET_PERSIST", "1").strip() != "0"


def _budget_dir() -> Path:
    global _BUDGET_DIR
    if _BUDGET_DIR is not None:
        return _BUDGET_DIR
    base = Path(os.getenv("EFFGEN_BUDGET_DIR", Path.home() / ".effgen" / "budget"))
    base.mkdir(parents=True, exist_ok=True)
    _BUDGET_DIR = base
    return _BUDGET_DIR


def _day_file(day: str) -> Path:
    return _budget_dir() / f"{day}.json"


def _load_day(day: str) -> dict[str, float]:
    """Return the ledger for *day*, loading from disk on first access."""
    if day in _ledger:
        return _ledger[day]
    data: dict[str, float] = {}
    if _persist_enabled():
        path = _day_file(day)
        if path.exists():
            try:
                data = {k: float(v) for k, v in json.loads(path.read_text()).items()}
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load budget ledger %s: %s", path, exc)
    _ledger[day] = data
    return data


def _save_day(day: str) -> None:
    if not _persist_enabled():
        return
    try:
        _day_file(day).write_text(json.dumps(_ledger.get(day, {})))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist budget ledger for %s: %s", day, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_spend(principal: str, *, day: str | None = None) -> float:
    """Return *principal*'s accrued spend (USD) for *day* (default today)."""
    day = day or _today()
    with _lock:
        return _load_day(day).get(principal, 0.0)


def check_budget(principal: str, cap: float, *, day: str | None = None) -> None:
    """Raise :class:`BudgetExceeded` if *principal* has already met *cap*.

    A *cap* of ``0.0`` means unlimited and never raises.

    Args:
        principal: Whose spend is checked.
        cap: The daily limit in US dollars; ``0.0`` means unlimited.
        day: The accounting day to check, defaulting to today.
    """
    if cap <= 0.0:
        return
    spent = get_spend(principal, day=day)
    if spent >= cap:
        raise BudgetExceeded(
            f"BudgetExceeded: principal {principal!r} has spent "
            f"${spent:.4f} of its ${cap:.2f}/day cap"
        )


def charge(
    principal: str,
    amount: float,
    *,
    cap: float = 0.0,
    day: str | None = None,
) -> float:
    """Add *amount* USD to *principal*'s spend for *day* and return new total.

    If *cap* is positive and the principal has **already** met or exceeded the
    cap before this charge, :class:`BudgetExceeded` is raised and nothing is
    charged. The charge itself may push spend past the cap (the next call then
    rejects), which matches the spec: a cap of $0.01 admits the first call and
    rejects subsequent ones.
    """
    day = day or _today()
    with _lock:
        ledger = _load_day(day)
        current = ledger.get(principal, 0.0)
        if cap > 0.0 and current >= cap:
            raise BudgetExceeded(
                f"BudgetExceeded: principal {principal!r} has spent "
                f"${current:.4f} of its ${cap:.2f}/day cap"
            )
        ledger[principal] = current + max(0.0, amount)
        _save_day(day)
        return ledger[principal]


def _reserved_total(principal: str, day: str) -> float:
    """Sum of in-flight reservations for *principal* on *day* (lock held)."""
    return sum(_reservations.get(day, {}).get(principal, {}).values())


def reserve(
    principal: str,
    amount: float,
    *,
    cap: float = 0.0,
    day: str | None = None,
) -> str:
    """Reserve *amount* USD against *principal*'s budget and return a token.

    The reservation counts against the cap immediately (alongside already
    committed spend) so an over-budget or concurrently-bursting principal is
    rejected *before* the request runs. Nothing is permanently charged until
    :func:`reconcile` is called; :func:`release` discards the reservation
    without charging (used when the underlying call fails).

    Raises :class:`BudgetExceeded` if committed + reserved spend has already met
    or exceeded a positive *cap*.

    Args:
        principal: Whose budget the reservation is taken from.
        amount: The estimated cost to hold, in US dollars.
        cap: The daily limit in US dollars; ``0.0`` means unlimited.
        day: The accounting day to reserve against, defaulting to today.
    """
    import uuid

    day = day or _today()
    with _lock:
        committed = _load_day(day).get(principal, 0.0)
        reserved = _reserved_total(principal, day)
        if cap > 0.0 and (committed + reserved) >= cap:
            raise BudgetExceeded(
                f"BudgetExceeded: principal {principal!r} has spent/reserved "
                f"${committed + reserved:.4f} of its ${cap:.2f}/day cap"
            )
        token = uuid.uuid4().hex
        _reservations.setdefault(day, {}).setdefault(principal, {})[token] = max(0.0, amount)
        return token


def _pop_reservation(principal: str, token: str, day: str) -> float:
    """Remove and return a reservation amount (lock held); 0.0 if missing."""
    by_principal = _reservations.get(day, {}).get(principal, {})
    return by_principal.pop(token, 0.0)


def reconcile(
    principal: str,
    token: str,
    actual_amount: float | None = None,
    *,
    day: str | None = None,
) -> float:
    """Commit a reservation, charging *actual_amount* (default = reserved).

    Returns the principal's new committed total. Pass ``actual_amount`` when the
    real provider usage/cost is known after the call completes; otherwise the
    originally reserved estimate is charged.

    Args:
        principal: Whose budget is charged.
        token: The reservation token to commit.
        actual_amount: What the call really cost, defaulting to the reservation.
        day: The accounting day to charge, defaulting to today.
    """
    day = day or _today()
    with _lock:
        reserved = _pop_reservation(principal, token, day)
        charge_amount = reserved if actual_amount is None else max(0.0, actual_amount)
        ledger = _load_day(day)
        ledger[principal] = ledger.get(principal, 0.0) + charge_amount
        _save_day(day)
        return ledger[principal]


def release(principal: str, token: str, *, day: str | None = None) -> None:
    """Discard a reservation without charging (e.g. the call failed).

    Args:
        principal: Whose reservation is discarded.
        token: The reservation token to discard.
        day: The accounting day the reservation was made on.
    """
    day = day or _today()
    with _lock:
        _pop_reservation(principal, token, day)


def reset(principal: str | None = None, *, day: str | None = None) -> None:
    """Reset spend. With no args, clears the entire in-memory ledger.

    Primarily for tests and admin tooling.
    """
    global _ledger, _BUDGET_DIR, _reservations
    with _lock:
        if principal is None and day is None:
            _ledger = {}
            _reservations = {}
            return
        d = day or _today()
        ledger = _load_day(d)
        if principal is None:
            ledger.clear()
        else:
            ledger.pop(principal, None)
        _save_day(d)
