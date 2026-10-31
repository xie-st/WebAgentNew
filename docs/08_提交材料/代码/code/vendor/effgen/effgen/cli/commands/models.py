"""The ``effgen models`` command group: list, browse, info, load, status, refresh.

``_main`` parses arguments and dispatches; the ``CLIInterface._models_*`` and
related methods delegate here. Holds the drift-aware catalog views, the local
HuggingFace-cache probing, price-cell formatting, and the live-refresh reporting.

Functions receive the ``CLIInterface`` as ``cli`` and reach registry helpers
through it (``cli._local_cached_models()`` etc.), so an instance override or a
test's ``monkeypatch.setattr(cli, ...)`` takes effect on the moved code too.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from effgen.cli.commands._shared import _print_group_help, resolve_provider_name
from effgen.ui.tables import check_mark

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface

logger = logging.getLogger(__name__)


def _coding_cell(rec: Any) -> str:
    """How a record's coding suitability reads in the ``models info`` table."""
    suitability = rec.coding
    if suitability.is_suitable:
        return "suitable"
    text = f"{suitability.verdict} — {suitability.reason}."
    if suitability.fix:
        text += f" {suitability.fix}"
    if suitability.measured_on:
        text += f" (measured {suitability.measured_on})"
    return text


def models_commands(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Model management commands.

    Args:
        args: Parsed command-line arguments
    """
    if args.model_command == 'list':
        return cli._models_list(args) or 0
    elif args.model_command == 'browse':
        return cli._models_browse(args) or 0
    elif args.model_command == 'info':
        return cli._models_info(args) or 0
    elif args.model_command == 'load':
        cli._models_load(args)
    elif args.model_command == 'unload':
        cli._models_unload(args)
    elif args.model_command == 'status':
        cli._models_status(args)
    elif args.model_command == 'refresh':
        return cli._models_refresh(args) or 0
    elif args.model_command is None:
        return _print_group_help(args)
    else:
        cli.print_error(f"Unknown models command: {args.model_command}")
        return 1

    return 0


def price_cell(rec: Any) -> str:
    """Format a model's input/output price per 1M tokens for a table cell."""
    pin, pout = rec.price_in_per_1m, rec.price_out_per_1m
    # A genuinely nonzero published rate is shown as-is (mirrors
    # ``_catalog_pricing`` in ``effgen.models._cost``, the single source of
    # truth for how a $0 row is labeled).
    if (pin or 0) > 0 or (pout or 0) > 0:
        fmt = lambda v: ("?" if v is None else f"${v:g}")  # noqa: E731
        return f"{fmt(pin)}/{fmt(pout)}"
    # No nonzero rate: a genuine free tier reads "free"; a non-token billing
    # note reads "metered"; anything else (including an explicit 0/0 with no
    # free-tier flag) has no published price and reads "unpriced" rather than
    # a fabricated "$0".
    if rec.free_tier:
        return "free"
    if rec.price_note:
        return "metered"
    return "unpriced"


def local_cached_models(cli: "CLIInterface") -> list[dict]:
    """Models actually downloaded in the local HuggingFace cache (on disk).

    Each entry carries a ``complete`` flag: a snapshot with no real weight
    files (only an interrupted download, e.g. ``.incomplete`` blobs plus a
    shard manifest) is reported as incomplete so it isn't mistaken for ready.
    """
    out: list[dict] = []
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        for repo in sorted(info.repos, key=lambda r: r.repo_id):
            if repo.repo_type != "model":
                continue
            weight_files = {
                f.file_name
                for rev in repo.revisions
                for f in rev.files
                if f.file_name.endswith(cli._WEIGHT_SUFFIXES)
                and not f.file_name.endswith(".index.json")
            }
            out.append({
                "id": repo.repo_id,
                "size_gb": repo.size_on_disk / (1024 ** 3),
                "path": str(repo.repo_path),
                "complete": bool(weight_files),
            })
    except Exception as e:  # noqa: BLE001 - cache scan is best-effort
        logger.debug(f"HF cache scan failed: {e}")
    return out


def local_model_context_window(cli: "CLIInterface", path: str) -> int | None:
    """Read the model's max context length from its on-disk ``config.json``."""
    import glob
    for cfg in glob.glob(os.path.join(path, "snapshots", "*", "config.json")):
        try:
            with open(cfg, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            val = data.get(key)
            if isinstance(val, int) and val > 0:
                return val
    return None


def models_list(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """List models from the drift-aware registry (not a static yaml).

    Shows three views — the provider registry (the bundled, refreshable
    catalog), the local HuggingFace cache (what's actually downloaded), and
    a per-provider summary with auth readiness and the snapshot's
    "verified on" date so users can tell when the data was last confirmed.
    """
    from effgen.models import _catalog, _refresh

    provider_filter, prov_err = resolve_provider_name(getattr(args, "provider", None))
    if prov_err:
        cli.print_error(prov_err)
        return 1
    free_only = bool(getattr(args, "free", False))
    tools_only = bool(getattr(args, "tools", False))

    providers = [provider_filter] if provider_filter else list(_catalog.known_providers())

    def _records(prov: str):
        recs = _catalog.list_models(prov)
        if free_only:
            recs = [r for r in recs if r.free_tier]
        if tools_only:
            recs = [r for r in recs if r.supports_tools]
        return recs

    # ---- JSON output ----------------------------------------------------
    if getattr(args, "output_json", False):
        payload: dict[str, Any] = {"providers": {}, "local_cache": cli._local_cached_models()}
        for prov in providers:
            meta = _catalog.snapshot_meta(prov)
            payload["providers"][prov] = {
                "verified_on": meta.get("verified_on"),
                "count": len(_catalog.list_models(prov)),
                "default_model": _catalog.default_model(prov),
                "auth_ready": _refresh.has_credentials(prov),
                "models": [
                    {
                        "id": r.id, "family": r.family,
                        "context_window": r.context_window,
                        "max_output": r.max_output,
                        "price_in_per_1m": r.price_in_per_1m,
                        "price_out_per_1m": r.price_out_per_1m,
                        "supports_tools": r.supports_tools,
                        "supports_vision": r.supports_vision,
                        "supports_audio": r.supports_audio,
                        "free_tier": r.free_tier, "deprecated": r.deprecated,
                        "is_priced": r.is_priced,
                        "coding": r.coding.verdict,
                        "price_source": r.price_source,
                    }
                    for r in _records(prov)
                ],
            }
        print(json.dumps(payload, indent=2))
        return 0

    # ---- Provider registry view ----------------------------------------
    cli.print_header("Available Models")

    if provider_filter:
        # Full per-model detail for one provider.
        recs = _records(provider_filter)
        meta = _catalog.snapshot_meta(provider_filter)
        auth = "ready" if _refresh.has_credentials(provider_filter) else "no key"
        verified = meta.get("verified_on") or "unknown"
        default_id = _catalog.default_model(provider_filter)
        if cli._rich_tables():
            title = (f"{provider_filter} — {len(recs)} models "
                     f"(auth: {auth}, verified: {verified})")
            from rich.table import Table
            table = Table(title=title)
            table.add_column("Model ID", style="effgen.model", overflow="fold")
            table.add_column("Context", justify="right", no_wrap=True)
            table.add_column("Max Out", justify="right", no_wrap=True)
            table.add_column("$/1M in/out", style="effgen.cost", no_wrap=True, overflow="fold")
            table.add_column("Tools", justify="center", no_wrap=True)
            table.add_column("Vision", justify="center", no_wrap=True)
            table.add_column("Free", justify="center", no_wrap=True)
            table.add_column("Status", style="effgen.warning")
            for r in recs:
                status = "deprecated" if r.deprecated else ("default" if r.id == default_id else "")
                table.add_row(
                    r.id,
                    f"{r.context_window:,}" if r.context_window else "—",
                    f"{r.max_output:,}" if r.max_output else "—",
                    cli._price_cell(r),
                    check_mark(r.supports_tools),
                    check_mark(r.supports_vision),
                    check_mark(r.free_tier),
                    status,
                )
            cli.console.print(table)
            cli.console.print(
                f"\n[dim]Pricing source: catalog snapshot. "
                f"Run [cyan]effgen models refresh --provider {provider_filter}[/cyan] "
                f"to update from the live API.[/dim]",
                highlight=False,
            )
        else:
            id_w = min(max((len(r.id) for r in recs), default=8), 60)
            print(f"{provider_filter} — {len(recs)} models "
                  f"(auth: {auth}, verified: {verified})")
            for r in recs:
                pin, pout = cli._price_in_out_cells(r)
                mark = " *" if r.id == default_id else ""
                print(f"  {r.id:<{id_w}}  ctx={r.context_window or '-':>9}  "
                      f"in={pin:>9}  out={pout:>9}  "
                      f"{'tools' if r.supports_tools else '':<5} "
                      f"{'vision' if r.supports_vision else ''}{mark}")
        return 0

    # Overview: per-provider summary + filtered flat table when filtering.
    if free_only or tools_only:
        label = "free-tier" if free_only else "tool-capable"
        if tools_only and free_only:
            label = "free + tool-capable"
        if cli._rich_tables():
            from rich.table import Table
            table = Table(title=f"{label.capitalize()} models (all providers)")
            table.add_column("Model ID", style="effgen.model", overflow="fold")
            table.add_column("Provider", style="effgen.accent", no_wrap=True)
            table.add_column("Context", justify="right", no_wrap=True)
            table.add_column("$/1M in/out", style="effgen.cost", no_wrap=True, overflow="fold")
            table.add_column("Tools", justify="center", no_wrap=True)
            table.add_column("Free", justify="center", no_wrap=True)
            for prov in providers:
                for r in _records(prov):
                    table.add_row(
                        r.id, prov,
                        f"{r.context_window:,}" if r.context_window else "—",
                        cli._price_cell(r),
                        check_mark(r.supports_tools),
                        check_mark(r.free_tier),
                    )
            cli.console.print(table)
        else:
            flat = [r for prov in providers for r in _records(prov)]
            id_w = min(max((len(r.id) for r in flat), default=8), 60)
            for r in flat:
                print(f"{r.id:<{id_w}}  {r.provider:<10}  "
                      f"ctx={r.context_window or '-':>9}  {cli._price_cell(r)}")
        return 0

    # Default overview: one row per provider.
    stale = set(_catalog.stale_providers())
    if cli.console:
        from rich.table import Table
        table = Table(title="Provider Registry (bundled catalog)")
        table.add_column("Provider", style="effgen.accent")
        table.add_column("Models", justify="right")
        table.add_column("Default", style="effgen.model", overflow="fold")
        table.add_column("Auth", justify="center")
        table.add_column("Verified", style="effgen.muted")
        for prov in providers:
            meta = _catalog.snapshot_meta(prov)
            n = len(_catalog.list_models(prov))
            auth = ("[effgen.success]key[/effgen.success]" if _refresh.has_credentials(prov)
                    else "[effgen.muted]—[/effgen.muted]")
            verified = meta.get("verified_on") or "?"
            if prov in stale:
                verified += " (stale)"
            table.add_row(prov, str(n), _catalog.default_model(prov) or "—", auth, verified)
        cli.console.print(table)
        cli.console.print(
            "\n[dim]Detail: [cyan]effgen models list --provider <name>[/cyan]  ·  "
            "Filter: [cyan]--free[/cyan] / [cyan]--tools[/cyan]  ·  "
            "Update: [cyan]effgen models refresh[/cyan][/dim]",
            highlight=False,
        )
    else:
        for prov in providers:
            n = len(_catalog.list_models(prov))
            auth = "key" if _refresh.has_credentials(prov) else "-"
            print(f"{prov:12s} {n:>4} models  default={_catalog.default_model(prov)}  auth={auth}")

    # ---- Local HuggingFace cache view ----------------------------------
    local = cli._local_cached_models()
    if local:
        n_ready = sum(1 for m in local if m.get("complete", True))
        if cli.console:
            from rich.table import Table
            ltable = Table(title=f"Local HuggingFace cache ({n_ready} ready)")
            ltable.add_column("Model", style="effgen.model", overflow="fold")
            ltable.add_column("Size", justify="right")
            ltable.add_column("Status", justify="center")
            for m in local:
                ready = m.get("complete", True)
                ltable.add_row(
                    m["id"], f"{m['size_gb']:.1f} GB",
                    "[effgen.success]ready[/effgen.success]" if ready
                    else "[effgen.warning]incomplete[/effgen.warning]",
                )
            cli.console.print(ltable)
        else:
            print("\nLocal HuggingFace cache:")
            for m in local:
                tag = "" if m.get("complete", True) else "  (incomplete)"
                print(f"  {m['id']}  ({m['size_gb']:.1f} GB){tag}")
    return 0


def rich_tables(cli: "CLIInterface") -> bool:
    """True when rich table rendering fits the destination (a real terminal).

    Piped or redirected output narrows to a default width that truncates or
    drops columns; there the catalog views emit complete, aligned plain text
    instead so no model id or price is lost.
    """
    return bool(cli.console) and bool(getattr(cli.console, "is_terminal", False))


def browse_filter_sort(recs: list[Any], args: argparse.Namespace) -> list[Any]:
    """Apply the browse filters/sort to a list of catalog records.

    Filters compose (a record must satisfy every one supplied); records with
    no published input/output price are excluded by a ``--max-price-*``
    ceiling rather than treated as free. Returns the filtered, sorted list.
    """
    search = (getattr(args, "search", None) or "").lower().strip()
    min_ctx = getattr(args, "min_context", None)
    max_pin = getattr(args, "max_price_in", None)
    max_pout = getattr(args, "max_price_out", None)

    def keep(r) -> bool:
        if getattr(args, "free", False) and not r.free_tier:
            return False
        if getattr(args, "tools", False) and not r.supports_tools:
            return False
        if getattr(args, "vision", False) and not r.supports_vision:
            return False
        if getattr(args, "audio", False) and not r.supports_audio:
            return False
        if min_ctx is not None and (r.context_window or 0) < min_ctx:
            return False
        if max_pin is not None and (r.price_in_per_1m is None or r.price_in_per_1m > max_pin):
            return False
        if max_pout is not None and (r.price_out_per_1m is None or r.price_out_per_1m > max_pout):
            return False
        if search and search not in (
            f"{r.id} {r.family} {r.provider}".lower()
        ):
            return False
        return True

    out = [r for r in recs if keep(r)]

    sort = getattr(args, "sort", "provider") or "provider"
    # A missing numeric value sorts last on an ascending sort (unknown price
    # or context is worst-case), so it never masquerades as the cheapest.
    big = float("inf")

    def price_in(r):
        return r.price_in_per_1m if r.price_in_per_1m is not None else big

    def price_out(r):
        return r.price_out_per_1m if r.price_out_per_1m is not None else big

    keyers = {
        "provider": lambda r: (r.provider, r.id.lower()),
        "id": lambda r: r.id.lower(),
        "context": lambda r: (r.context_window or 0, r.id.lower()),
        "max-out": lambda r: (r.max_output or 0, r.id.lower()),
        "price-in": lambda r: (price_in(r), r.id.lower()),
        "price-out": lambda r: (price_out(r), r.id.lower()),
    }
    out.sort(key=keyers.get(sort, keyers["provider"]))
    if getattr(args, "desc", False):
        out.reverse()
    return out


def models_browse(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Browse the full cross-provider catalog with search/filter/sort/paging.

    Reads the bundled, refreshable catalog (the same source ``models list``
    and ``models info`` use). Every provider's models appear in one table so
    a single view answers "cheapest vision model over 128k context" without
    leaving the terminal. Price labeling is exact: an unpriced row reads
    ``unpriced``, a free tier reads ``free``.
    """
    from effgen.models import _catalog

    provider_filter, prov_err = resolve_provider_name(getattr(args, "provider", None))
    if prov_err:
        cli.print_error(prov_err)
        return 1

    recs = _catalog.list_models(provider_filter)
    matched = cli._browse_filter_sort(recs, args)
    total = len(matched)

    offset = max(0, getattr(args, "offset", 0) or 0)
    limit = getattr(args, "limit", None)
    page = matched[offset:offset + limit] if limit else matched[offset:]

    include_local = bool(getattr(args, "include_local", False))
    local = cli._local_cached_models() if include_local else []

    # The snapshot "verified on" date is per provider, not stamped on the
    # bundled record; resolve it once per provider on the page so the JSON
    # provenance field carries the same date the table footer and the
    # dashboard show, rather than a null.
    verified_by_provider: dict[str, str | None] = {}

    def _verified_on(prov: str) -> str | None:
        if prov not in verified_by_provider:
            verified_by_provider[prov] = _catalog.snapshot_meta(prov).get("verified_on")
        return verified_by_provider[prov]

    # ---- JSON output ----------------------------------------------------
    if getattr(args, "output_json", False):
        payload: dict[str, Any] = {
            "count": total,
            "offset": offset,
            "limit": limit,
            "models": [
                {
                    "id": r.id, "provider": r.provider, "family": r.family,
                    "context_window": r.context_window, "max_output": r.max_output,
                    "price_in_per_1m": r.price_in_per_1m,
                    "price_out_per_1m": r.price_out_per_1m,
                    "supports_tools": r.supports_tools,
                    "supports_vision": r.supports_vision,
                    "supports_audio": r.supports_audio,
                    "free_tier": r.free_tier, "deprecated": r.deprecated,
                    "is_priced": r.is_priced,
                    "coding": r.coding.verdict,
                    "price_source": r.price_source,
                    "verified_on": r.verified_on or _verified_on(r.provider),
                }
                for r in page
            ],
        }
        if include_local:
            payload["local_cache"] = local
        print(json.dumps(payload, indent=2))
        return 0

    # ---- Human table ----------------------------------------------------
    cli.print_header("Model Catalog")
    if not matched:
        cli.print("No models match those filters. Loosen a filter or run "
                  "[cyan]effgen models browse[/cyan] with no filters." if cli.console
                  else "No models match those filters.")
        return 0

    # The cross-provider table carries nine columns; on a narrow terminal
    # rich would starve the (foldable) Model ID column — the one field this
    # view exists for — to keep the fixed numeric columns, folding or even
    # hiding the id. Below this width the complete aligned plain-text table
    # reads better and never drops an id or a price.
    wide_enough = getattr(cli.console, "width", 0) >= 100
    if cli._rich_tables() and wide_enough:
        shown = f"showing {len(page)} of {total}"
        if offset:
            shown += f" (from #{offset + 1})"
        from rich.table import Table
        table = Table(title=f"Models across providers — {shown}")
        table.add_column("Provider", style="effgen.accent", no_wrap=True)
        table.add_column("Model ID", style="effgen.model", overflow="fold")
        table.add_column("Context", justify="right", no_wrap=True)
        table.add_column("Max Out", justify="right", no_wrap=True)
        table.add_column("$/1M in", style="effgen.cost", justify="right",
                         no_wrap=True, overflow="fold")
        table.add_column("$/1M out", style="effgen.cost", justify="right",
                         no_wrap=True, overflow="fold")
        table.add_column("Tools", justify="center", no_wrap=True)
        table.add_column("Vision", justify="center", no_wrap=True)
        table.add_column("Free", justify="center", no_wrap=True)
        for r in page:
            pin, pout = cli._price_in_out_cells(r)
            table.add_row(
                r.provider, r.id,
                f"{r.context_window:,}" if r.context_window else "—",
                f"{r.max_output:,}" if r.max_output else "—",
                pin, pout,
                check_mark(r.supports_tools),
                check_mark(r.supports_vision),
                check_mark(r.free_tier),
            )
        cli.console.print(table)
        if limit and offset + limit < total:
            cli.console.print(
                f"\n[dim]More: [cyan]--offset {offset + limit}[/cyan] "
                f"for the next page.[/dim]",
                highlight=False,
            )
        cli.console.print(
            "\n[dim]Pricing source: catalog snapshot. "
            "Update: [cyan]effgen models refresh[/cyan]  ·  "
            "Detail: [cyan]effgen models info <id>[/cyan][/dim]",
            highlight=False,
        )
    else:
        # Complete, aligned plain text for piped/redirected output — every
        # model id and price in full, no width-driven truncation.
        id_w = max((len(r.id) for r in page), default=8)
        id_w = min(max(id_w, 8), 60)
        prov_w = max((len(r.provider) for r in page), default=8)
        header = (f"{'PROVIDER':<{prov_w}}  {'MODEL ID':<{id_w}}  "
                  f"{'CONTEXT':>9}  {'MAXOUT':>7}  {'$/1M IN':>9}  "
                  f"{'$/1M OUT':>9}  TOOLS  VIS  FREE")
        print(header)
        for r in page:
            pin, pout = cli._price_in_out_cells(r)
            print(f"{r.provider:<{prov_w}}  {r.id:<{id_w}}  "
                  f"{(f'{r.context_window:,}' if r.context_window else '-'):>9}  "
                  f"{(f'{r.max_output:,}' if r.max_output else '-'):>7}  "
                  f"{pin:>9}  {pout:>9}  "
                  f"{'yes' if r.supports_tools else '-':>5}  "
                  f"{'yes' if r.supports_vision else '-':>3}  "
                  f"{'yes' if r.free_tier else '-':>4}")
        print(f"\nshowing {len(page)} of {total}"
              + (f" (from #{offset + 1})" if offset else "")
              + "  ·  pricing from catalog snapshot")

    if include_local and local:
        if cli.console:
            cli.console.print(f"\n[dim]Local cache: {len(local)} model(s) "
                              f"— [cyan]effgen models list[/cyan] for detail.[/dim]", highlight=False)
        else:
            print("\nLocal cache:")
            for m in local:
                print(f"  {m['id']}  ({m['size_gb']:.1f} GB)")
    return 0


def price_in_out_cells(rec: Any) -> tuple[str, str]:
    """Return (input, output) price cells for the split-column browse table.

    A published nonzero rate shows as ``$<n>``; a genuine free tier reads
    ``free``, non-token billing reads ``metered``, and an unknown rate reads
    ``unpriced`` — never a fabricated ``$0`` (mirrors :func:`price_cell`).
    """
    pin, pout = rec.price_in_per_1m, rec.price_out_per_1m
    if (pin or 0) > 0 or (pout or 0) > 0:
        fmt = lambda v: ("?" if v is None else f"${v:g}")  # noqa: E731
        return fmt(pin), fmt(pout)
    label = "free" if rec.free_tier else ("metered" if rec.price_note else "unpriced")
    return label, label


def local_model_payload(cli: "CLIInterface", entry: dict) -> dict:
    """Build the local-cache facts for one model: engines, size, ctx, status."""
    import importlib.util
    engines = ["transformers"]
    if importlib.util.find_spec("vllm") is not None:
        engines.append("vllm")
    return {
        "id": entry["id"],
        "cached": True,
        "complete": entry.get("complete", True),
        "size_gb": entry["size_gb"],
        "path": entry.get("path"),
        "context_window": cli._local_model_context_window(entry.get("path", "")),
        "engines": engines,
    }


def render_local_model_info(cli: "CLIInterface", payload: dict) -> None:
    """Render the 'this model is in your local cache' block for `models info`."""
    ctx = payload.get("context_window")
    status = "ready" if payload.get("complete", True) else "incomplete download"
    rows = {
        "Local copy": "yes (HuggingFace cache)",
        "Status": status,
        "On-disk size": f"{payload['size_gb']:.1f} GB",
        "Local engines": ", ".join(payload["engines"]),
        "Context window": f"{ctx:,}" if ctx else "—",
    }
    run_hint = (f"effgen run -m {payload['id']} --engine transformers \"...\"")
    if cli.console:
        from rich.table import Table
        table = Table(show_header=False, title="Local cache")
        table.add_column("Field", style="effgen.label")
        table.add_column("Value", overflow="fold")
        for k, v in rows.items():
            table.add_row(k, str(v))
        cli.console.print(table)
        cli.console.print(f"\n[dim]Run locally: [cyan]{run_hint}[/cyan]"
                          f"  ·  or [cyan]load_model(\"{payload['id']}\", "
                          f"engine=\"transformers\")[/cyan][/dim]", highlight=False)
    else:
        print("\nLocal cache:")
        for k, v in rows.items():
            print(f"  {k}: {v}")
        print(f"  Run locally: {run_hint}")


def models_info(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Show detailed information for one model from the registry."""
    if not args.name:
        cli.print_error("Model name required")
        return 1

    from effgen.models import _catalog, _refresh
    from effgen.models.model_loader import ModelLoader

    # An engine-prefixed id (e.g. "transformers:Qwen/Qwen2.5-1.5B-Instruct")
    # names a local engine + a bare repo id. Strip the prefix so the id
    # matches the local cache and the catalog, and remember the engine so we
    # can lead with the local view.
    lookup_name = args.name
    requested_engine = None
    if ":" in lookup_name:
        _prefix, _rest = lookup_name.split(":", 1)
        if _prefix in ModelLoader._LOCAL_ENGINE_PREFIXES and _rest:
            requested_engine = _prefix
            lookup_name = _rest

    # Is this id sitting in the local HF cache? If so we can describe it as
    # locally-runnable even when the cloud catalog has no (or a different) row.
    local_entry = next(
        (m for m in cli._local_cached_models() if m["id"] == lookup_name), None
    )

    # An explicit local-engine request is answered from the local cache: show
    # the cached copy, or report a cache miss naming what is cached (rather
    # than cloud catalog suggestions the engine can't run).
    if requested_engine is not None:
        if local_entry is not None:
            local_payload = cli._local_model_payload(local_entry)
            if getattr(args, "output_json", False):
                print(json.dumps({"id": lookup_name, "provider": None,
                                   "engine": requested_engine,
                                   "local": local_payload}, indent=2))
                return 0
            cli.print_header(f"Model: {lookup_name} ({requested_engine})")
            cli._render_local_model_info(local_payload)
            return 0
        cached = [m["id"] for m in cli._local_cached_models()]
        if getattr(args, "output_json", False):
            print(json.dumps({"id": lookup_name, "provider": None,
                               "engine": requested_engine, "local": None,
                               "cached_models": cached}, indent=2))
            return 1
        cli.print_error(
            f"Model '{lookup_name}' is not in the local cache, so the "
            f"'{requested_engine}' engine can't run it yet."
        )
        if cached:
            cli.print("Locally cached models:")
            for cid in cached:
                cli.print(f"  {cid}")
            cli.print(f"\nDownload it first: effgen run -m {lookup_name} "
                      f"--engine {requested_engine} \"...\" (with network access).")
        return 1

    rec = _catalog.lookup(lookup_name)
    if rec is None:
        if local_entry is not None:
            # Downloaded locally but not in the cloud catalog: describe the local
            # copy instead of a misleading "not found in catalog".
            local_payload = cli._local_model_payload(local_entry)
            if getattr(args, "output_json", False):
                print(json.dumps({"id": args.name, "provider": None,
                                   "local": local_payload}, indent=2))
                return 0
            cli.print_header(f"Model: {args.name} (local)")
            cli._render_local_model_info(local_payload)
            return 0
        # Helpful not-found: suggest the nearest catalog ids + provider:id form.
        cli.print_error(f"Model not found in catalog: {args.name}")
        alts = _catalog.nearest_alternatives(args.name, n=5)
        if alts:
            cli.print("Did you mean:")
            for a in alts:
                cli.print(f"  {a.provider}:{a.id}")
        cli.print("\nRun 'effgen models list' to browse the catalog, or "
                  "'effgen models refresh' to update it.")
        return 1

    # The same model id can be served by more than one provider at different
    # prices; surface every alternative so the choice of *where* to run it is
    # visible (the resolved provider stays first).
    others = [v for v in _catalog.variants(rec.id) if v.provider != rec.provider]

    if getattr(args, "output_json", False):
        print(json.dumps({
            "id": rec.id, "provider": rec.provider, "display_name": rec.display_name,
            "family": rec.family, "context_window": rec.context_window,
            "max_output": rec.max_output, "price_in_per_1m": rec.price_in_per_1m,
            "price_out_per_1m": rec.price_out_per_1m, "supports_tools": rec.supports_tools,
            "supports_vision": rec.supports_vision, "supports_audio": rec.supports_audio,
            "free_tier": rec.free_tier, "deprecated": rec.deprecated,
            "rpm": rec.rpm, "tpm": rec.tpm, "rpd": rec.rpd,
            "price_source": rec.price_source, "verified_on": rec.verified_on,
            "notes": rec.notes, "coding": rec.coding.to_dict(),
            "also_available": [
                {
                    "provider": v.provider,
                    "price_in_per_1m": v.price_in_per_1m,
                    "price_out_per_1m": v.price_out_per_1m,
                    "context_window": v.context_window,
                    "supports_tools": v.supports_tools,
                    "supports_vision": v.supports_vision,
                    "free_tier": v.free_tier,
                }
                for v in others
            ],
            "local": cli._local_model_payload(local_entry) if local_entry else None,
        }, indent=2))
        return 0

    cli.print_header(f"Model: {rec.provider}:{rec.id}")
    meta = _catalog.snapshot_meta(rec.provider)
    rows = {
        "Provider": rec.provider,
        "Display name": rec.display_name or rec.id,
        "Family": rec.family or "—",
        "Context window": f"{rec.context_window:,}" if rec.context_window else "—",
        "Max output": f"{rec.max_output:,}" if rec.max_output else "—",
        "Price ($/1M in / out)": cli._price_cell(rec),
        "Tool calling": "yes" if rec.supports_tools else "no",
        "Coding": _coding_cell(rec),
        "Vision": "yes" if rec.supports_vision else "no",
        "Audio": "yes" if rec.supports_audio else "no",
        "Free tier": "yes" if rec.free_tier else "no",
        "Rate limits (rpm/tpm/rpd)": f"{rec.rpm or '—'} / {rec.tpm or '—'} / {rec.rpd or '—'}",
        "Deprecated": "yes" if rec.deprecated else "no",
        "Price source": rec.price_source,
        "Verified on": rec.verified_on or meta.get("verified_on") or "unknown",
        "Auth ready": "yes" if _refresh.has_credentials(rec.provider) else "no (set key)",
    }
    if cli.console:
        from rich.table import Table
        table = Table(show_header=False)
        table.add_column("Field", style="effgen.label")
        table.add_column("Value", overflow="fold")
        for k, v in rows.items():
            table.add_row(k, str(v))
        cli.console.print(table)
    else:
        for k, v in rows.items():
            print(f"  {k}: {v}")

    # When several providers serve this id, compare them so the analyst can
    # pick where to run it (pin the choice with a ``provider:id`` form).
    if others:
        if cli.console:
            from rich.table import Table
            vtable = Table(title=f"Also served by ({len(others)} other provider(s))")
            vtable.add_column("Provider", style="effgen.accent", no_wrap=True)
            vtable.add_column("$/1M in/out", style="effgen.cost", no_wrap=True, overflow="fold")
            vtable.add_column("Context", justify="right", no_wrap=True)
            vtable.add_column("Tools", justify="center", no_wrap=True)
            vtable.add_column("Vision", justify="center", no_wrap=True)
            vtable.add_column("Free", justify="center", no_wrap=True)
            for v in others:
                vtable.add_row(
                    v.provider, cli._price_cell(v),
                    f"{v.context_window:,}" if v.context_window else "—",
                    check_mark(v.supports_tools),
                    check_mark(v.supports_vision),
                    check_mark(v.free_tier),
                )
            cli.console.print(vtable)
            cli.console.print(
                f"\n[dim]Pin a provider: "
                f"[cyan]effgen run --provider <name> -m {rec.id}[/cyan][/dim]",
                highlight=False,
            )
        else:
            names = ", ".join(v.provider for v in others)
            print(f"\n  Also served by: {names} "
                  f"(pin with 'provider:{rec.id}')")

    cloud_hint = f"effgen run --provider {rec.provider} -m {rec.id} \"...\""
    # If the same id is also downloaded locally, lead with the local engine
    # path (the local block prints its own "Run locally" hint) and show the
    # cloud invocation as an alternative — don't present it as cloud-only.
    if local_entry is not None:
        cli._render_local_model_info(cli._local_model_payload(local_entry))
        if cli.console:
            cli.console.print(f"\n[dim]Cloud alternative: [cyan]{cloud_hint}[/cyan][/dim]", highlight=False)
        else:
            print(f"\n  Cloud alternative: {cloud_hint}")
    elif cli.console:
        cli.console.print(f"\n[dim]Use: [cyan]{cloud_hint}[/cyan][/dim]", highlight=False)
    return 0


def models_load(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Pre-load a model into the model pool."""
    from effgen.models.pool import ModelPool

    model_name = args.name
    engine = getattr(args, 'engine', None)
    cli.print(f"Loading model: {model_name}...")

    try:
        pool = ModelPool()
        pool.get_or_load(model_name, engine=engine)
        cli.print_success(f"Model '{model_name}' loaded successfully")

        # Show status
        for entry in pool.status():
            if entry["model_name"] == model_name:
                cli.print(f"  GPU memory: ~{entry['gpu_memory_gb']:.1f} GB")
    except Exception as e:
        cli.print_error(f"Failed to load model: {e}")
        return 1
    return None


def models_unload(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Unload a model from memory."""
    from effgen.models.model_loader import ModelLoader

    model_name = args.name
    cli.print(f"Unloading model: {model_name}...")

    try:
        loader = ModelLoader()
        if model_name in loader.loaded_models:
            loader.unload_model(model_name)
            cli.print_success(f"Model '{model_name}' unloaded")
        else:
            cli.print_warning(f"Model '{model_name}' is not currently loaded")
    except Exception as e:
        cli.print_error(f"Failed to unload model: {e}")
        return 1
    return None


def models_status(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Show loaded models and GPU memory status."""
    if getattr(args, "output_json", False):
        return cli._models_status_json()

    cli.print_header("Model & GPU Status")

    # GPU memory info — physical (driver) view across all processes, so this
    # reflects which GPUs are actually free, not just this process's usage.
    try:
        from effgen.gpu.cuda_compat import per_gpu_status
        gpus = per_gpu_status()
        gib = 1024 ** 3
        if gpus:
            if cli.console:
                from rich.table import Table
                gpu_table = Table(title="GPU Status (physical, all processes)")
                gpu_table.add_column("GPU", style="cyan")
                gpu_table.add_column("Name", style="white")
                gpu_table.add_column("Total", style="white")
                gpu_table.add_column("Used", style="yellow")
                gpu_table.add_column("Free", style="green")
                gpu_table.add_column("Util", justify="right")

                for g in gpus:
                    util = f"{g.utilization_pct:.0f}%" if g.utilization_pct is not None else "—"
                    gpu_table.add_row(
                        str(g.index), g.name,
                        f"{g.total_bytes / gib:.1f} GB",
                        f"{g.used_bytes / gib:.1f} GB",
                        f"{g.free_bytes / gib:.1f} GB",
                        util,
                    )
                cli.console.print(gpu_table)
            else:
                for g in gpus:
                    util = f", {g.utilization_pct:.0f}% util" if g.utilization_pct is not None else ""
                    print(f"GPU {g.index}: {g.name} — "
                          f"{g.total_bytes / gib:.1f} GB total, "
                          f"{g.used_bytes / gib:.1f} GB used, "
                          f"{g.free_bytes / gib:.1f} GB free{util}")
        else:
            try:
                import torch
                if not torch.cuda.is_available():
                    cli.print_warning("CUDA not available")
                else:
                    cli.print_warning("Could not query GPU memory status")
            except ImportError:
                cli.print_warning("PyTorch not installed — cannot query GPU status")
    except ImportError:
        cli.print_warning("PyTorch not installed — cannot query GPU status")

    # Loaded models
    from effgen.models.model_loader import ModelLoader
    loader = ModelLoader()
    loaded = loader.get_loaded_models()

    if loaded:
        cli.print("")
        cli.print_header("Loaded Models")
        for name, model in loaded.items():
            status = "loaded" if model.is_loaded() else "unloaded"
            cli.print(f"  {name}: {status}")
    else:
        cli.print("\nNo models currently loaded in this process.")

    # Capability registry
    from effgen.models.capabilities import list_registered_models
    registered = list_registered_models()
    cli.print(f"\nCapability profiles registered: {len(registered)}")
    return None


def models_status_json(cli: "CLIInterface") -> int:
    """Emit the GPU table + loaded models as JSON for ops/edge tooling."""
    gib = 1024 ** 3
    gpu_list: list[dict] = []
    cuda_available = True
    try:
        from effgen.gpu.cuda_compat import per_gpu_status
        for g in per_gpu_status():
            gpu_list.append({
                "index": g.index,
                "name": g.name,
                "total_gb": round(g.total_bytes / gib, 3),
                "used_gb": round(g.used_bytes / gib, 3),
                "free_gb": round(g.free_bytes / gib, 3),
                "utilization_pct": g.utilization_pct,
            })
        if not gpu_list:
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except ImportError:
                cuda_available = False
    except ImportError:
        cuda_available = False

    from effgen.models.capabilities import list_registered_models
    from effgen.models.model_loader import ModelLoader
    loaded = ModelLoader().get_loaded_models()
    loaded_list = [
        {"name": name, "loaded": bool(model.is_loaded())}
        for name, model in loaded.items()
    ]
    payload = {
        "cuda_available": cuda_available,
        "gpus": gpu_list,
        "loaded_models": loaded_list,
        "capability_profiles": len(list_registered_models()),
    }
    print(json.dumps(payload, indent=2))
    return 0


def models_refresh(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Refresh the bundled model catalog from each provider's live API.

    Fetches the live model list for the requested provider(s), reports what
    was added / removed / changed versus the bundled snapshot, and (unless
    ``--dry-run``) updates the snapshot so later runs see the fresh list
    offline. Providers without a configured key are skipped with a note.
    """
    from effgen.models import _refresh

    requested = getattr(args, "provider", None)
    dry_run = bool(getattr(args, "dry_run", False))

    if requested:
        if requested not in _refresh.refreshable_providers():
            cli.print_error(
                f"Unknown provider '{requested}'. "
                f"Refreshable: {', '.join(_refresh.refreshable_providers())}"
            )
            return 1
        providers = [requested]
    else:
        providers = _refresh.refreshable_providers()

    cli.print_header("Refresh model catalog" + (" (dry run)" if dry_run else ""))
    any_done = False
    had_error = False
    for provider in providers:
        if not _refresh.has_credentials(provider) and provider != "hf":
            if requested:  # explicit request for a keyless provider is an error
                cli.print_error(f"No API key for '{provider}'.")
                had_error = True
            else:
                cli.print(f"  {provider}: skipped (no key)")
            continue
        try:
            rep = _refresh.refresh_models(provider, persist=not dry_run)
        except Exception as e:  # noqa: BLE001 - report per-provider, keep going
            cli.print_error(f"{provider}: refresh failed: {e}")
            had_error = True
            continue
        any_done = True
        diff = rep["diff"]
        n_add, n_rem, n_chg = (
            len(diff["added"]), len(diff["removed"]), len(diff["changed"])
        )
        verb = "would update" if dry_run else "updated"
        cli.print_success(
            f"{provider}: {rep['live_count']} live models "
            f"(+{n_add} / -{n_rem} / ~{n_chg} changed) — {verb} snapshot"
        )
        for mid in diff["added"][:10]:
            cli.print(f"    + {mid}")
        for mid in diff["removed"][:10]:
            cli.print(f"    - {mid}")

    if had_error:
        return 1
    if not any_done:
        cli.print_warning(
            "No providers refreshed. Set a provider API key, e.g. "
            "OPENAI_API_KEY / CEREBRAS_API_KEY / GROQ_API_KEY."
        )
    return 0
