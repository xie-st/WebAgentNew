"""Official-platform observation, artifact and action adapter.

The official client gives the Agent a request dictionary, not a live browser:
observations arrive as an accessibility tree (plus optional screenshot, HTML and
DOM metadata), and actions are returned as one text command for the server to
execute. This module converts those requests into the A1 protocol and converts a
validated :class:`WebAction` back into the frozen SDK's exact action grammar.

Boundaries enforced here:

* AX element numbers are parsed only from ``observation.text``. DOM ``nodeId`` /
  ``backendNodeId`` values are never action numbers.
* ``goto`` is limited to a same-origin URL already observed in this task; the
  harness never opens a URL itself, least of all the server-internal address.
* screenshot/HTML downloads call the frozen ``third_party_agent.browser`` API
  synchronously during the current step and verify byte size and SHA-256 against
  the observation metadata. DOM is read directly from ``observation.dom.root``.
* A missing or failed artifact is represented as unavailable with a reason and an
  ``artifact_error`` event; it never falls back to the preceding frame.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ..errors import ActionRejected, DeadlineExceeded, ProtocolError, RejectReason
from ..protocol import (
    ActionOp,
    Artifact,
    ArtifactSet,
    Capabilities,
    ElementRef,
    Feedback,
    FeedbackStatus,
    Observation,
    WebAction,
    validate_run_id,
)
from ..redaction import DEFAULT_REDACTOR, Redactor
from ..session import Session
from ..tools import resolve_target, same_origin

_AX_ELEMENT_RE = re.compile(r"(?m)^[ \t]*\[([0-9]{1,12})\][ \t]*(.*)$")
_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_DOM_LIMIT = 4 * 1024 * 1024


class OfficialBackendError(ProtocolError):
    """The official request cannot be represented as a harness observation."""


class OfficialActionFormatError(ProtocolError):
    """An action cannot be written in the frozen SDK's action grammar."""


class OfficialArtifactError(RuntimeError):
    """An artifact download or integrity check failed."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = str(reason)
        self.message = str(message)
        super().__init__(f"{self.reason}: {self.message}")


def parse_ax_elements(
    text: str,
    *,
    page_id: str,
    observation_id: str,
) -> dict[str, ElementRef]:
    """Extract the current AX action numbers from the numbered tree text.

    Only lines whose first non-space token is ``[number]`` are considered. This
    is the AX numbering carried by the official observation, not a DOM id.
    """

    if not isinstance(text, str):
        raise OfficialBackendError("observation.text must be a string")
    elements: dict[str, ElementRef] = {}
    for match in _AX_ELEMENT_RE.finditer(text):
        number = match.group(1)
        if number in elements:
            continue
        rest = match.group(2).strip()
        role = "generic"
        name = rest
        if rest and rest[0] not in "'\"":
            parts = rest.split(None, 1)
            role = parts[0].lower()
            name = parts[1] if len(parts) > 1 else ""
        name = name.strip()
        if len(name) >= 2 and name[0] == name[-1] and name[0] in "'\"":
            name = name[1:-1]
        elements[number] = ElementRef(
            page_id=page_id,
            observation_id=observation_id,
            number=number,
            role=role,
            name=name,
            localization={"official_ax_id": number},
        )
    return elements


class TextMode(str, Enum):
    """How free text is put on the official action wire.

    ``RAW`` sends the text unchanged (including line breaks). ``DECOMPOSE``
    forbids a line break inside one ``type`` action: the execution layer splits a
    multi-line proposal into ``type`` + ``press [Enter]`` sub-actions first.
    Neither mode escapes backslashes or brackets: the platform does not implement
    an unescape routine (batch C task 731 typed a literal ``\\n``), so escaping
    would silently change the text.
    """

    RAW = "raw"
    DECOMPOSE = "decompose"


DEFAULT_TEXT_MODE = TextMode.DECOMPOSE

# The platform executes an empty ``type`` field with submit as an error and sends
# no feedback (batch 2da7a209 task 605: ``type [815] [] [1]``). An empty text with
# submit means "press Enter on the focused field", so it is sent as the existing
# Enter command instead of an empty type. An empty text without submit has no
# page effect at all and is refused locally before anything is dispatched.
TYPE_SUBMIT_ENTER_COMMAND = "press [Enter]"
EMPTY_TYPE_CORRECTION = (
    "web_type needs non-empty text when submit is false; no action was sent. "
    "To submit the field's current content, call web_press with keys='Enter'. "
    "To replace existing content, press Ctrl+A and Backspace first, then type "
    "the new value."
)


def normalize_text_mode(value: str | TextMode) -> TextMode:
    if isinstance(value, TextMode):
        return value
    try:
        return TextMode(str(value))
    except ValueError as exc:
        raise OfficialActionFormatError(
            f"unknown official text_mode {value!r}; use 'raw' or 'decompose'"
        ) from exc


def validate_action_text(value: str, field: str) -> str:
    """Refuse only characters the action wire cannot carry; never rewrite.

    Line breaks, tabs and carriage returns are allowed so a real-page run can
    observe their true semantics; other C0 controls, DEL and lone UTF-16
    surrogates have no representable wire form and raise explicitly.
    """

    if not isinstance(value, str):
        raise OfficialActionFormatError(f"{field} must be a string")
    for char in value:
        code = ord(char)
        if 0xD800 <= code <= 0xDFFF:
            raise OfficialActionFormatError(f"{field} contains an unencodable UTF-16 surrogate")
        if code < 0x20 and char not in "\n\r\t":
            raise OfficialActionFormatError(
                f"{field} contains control character U+{code:04X}, which the action grammar "
                "cannot represent"
            )
        if code == 0x7F:
            raise OfficialActionFormatError(
                f"{field} contains control character U+007F, which the action grammar "
                "cannot represent"
            )
    return value


def _wire_text(value: str, field: str, text_mode: TextMode) -> str:
    if not isinstance(value, str):
        raise OfficialActionFormatError(f"{field} must be a string")
    validate_action_text(value, field)
    if text_mode is TextMode.DECOMPOSE and ("\n" in value or "\r" in value):
        raise OfficialActionFormatError(
            f"{field} contains a line break; text_mode=decompose sends it as separate "
            "type/press sub-actions, so it must be split before serialization"
        )
    return value


def _escape_url(value: str) -> str:
    if not isinstance(value, str):
        raise OfficialActionFormatError("goto.url must be a string")
    if not value:
        raise OfficialActionFormatError("goto.url must not be empty")
    for char in value:
        code = ord(char)
        if 0xD800 <= code <= 0xDFFF:
            raise OfficialActionFormatError("goto.url contains an unencodable surrogate")
        if char in "[]\\" or char.isspace() or code < 0x20 or code == 0x7F:
            raise OfficialActionFormatError(
                "goto.url contains a bracket, backslash, whitespace or control character "
                "that would break the action grammar; percent-encode it first"
            )
    return value


def serialize_stop(
    message: str,
    *,
    text_mode: TextMode | str = DEFAULT_TEXT_MODE,
) -> str:
    """Return the official ``stop [说明]`` command.

    A stop ends the task and cannot be split into sub-actions, so its message is
    kept to one line: line breaks become spaces. Brackets and backslashes are
    sent unchanged; the previous custom escape produced literal ``\\n`` text on
    the platform and is not used.
    """

    del text_mode  # one terminal control line in both modes
    text = validate_action_text(message or "task stopped", "stop.message")
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
    text = text or "task stopped"
    return f"stop [{text}]"


class OfficialBackend:
    """A1 backend facade for one official task.

    It never executes actions or fetches a page on its own. ``from_request`` turns
    the current SDK callback payload into an observation; ``serialize_action``
    turns the one action the loop selected into the text the platform executes.
    """

    backend_id = "official"

    def __init__(
        self,
        run_id: str,
        *,
        session: Session | None = None,
        evidence: Any | None = None,
        page_id: str = "official-page",
        browser_api: Any | None = None,
        download_artifacts: frozenset[str] | set[str] | tuple[str, ...] = frozenset({"screenshot"}),
        deadline_check: Any | None = None,
        redactor: Redactor | None = None,
        text_mode: TextMode | str = DEFAULT_TEXT_MODE,
    ) -> None:
        validate_run_id(run_id)
        if session is not None and not isinstance(session, Session):
            raise OfficialBackendError("session must be a webagent Session")
        self.run_id = run_id
        self.page_id = page_id
        self.text_mode = normalize_text_mode(text_mode)
        self.session = session
        self.evidence = evidence if evidence is not None else (
            session.evidence if session is not None else None
        )
        self.browser_api = browser_api
        downloads = frozenset(download_artifacts)
        unknown = downloads - {"screenshot", "html", "dom"}
        if unknown:
            raise ValueError(f"unknown artifact download kinds: {', '.join(sorted(unknown))}")
        self.download_artifacts = downloads
        self.deadline_check = deadline_check
        self.redactor = redactor or (session.redactor if session is not None else DEFAULT_REDACTOR)
        self._observation_counter = 0
        self._current: Observation | None = None
        self._current_raw: Mapping[str, Any] | None = None
        self._raw_by_observation: dict[str, Mapping[str, Any]] = {}
        self._observed_urls: list[str] = []
        self._loaded_artifacts: dict[tuple[str, str], Artifact] = {}
        self.artifact_errors: list[dict[str, Any]] = []
        self._closed = False

    # ---------------------------------------------------------------- read API
    @property
    def current_observation(self) -> Observation | None:
        return self._current

    @property
    def observed_urls(self) -> tuple[str, ...]:
        return tuple(self._observed_urls)

    @property
    def current_raw_observation(self) -> Mapping[str, Any] | None:
        return self._current_raw

    # ---------------------------------------------------------------- protocol
    def observe(self) -> Observation:
        raise OfficialBackendError(
            "official observations arrive in the SDK step request; call from_request(request)"
        )

    def execute(self, action: WebAction) -> Any:
        raise OfficialBackendError(
            "official actions are executed by the platform; call serialize_action(action)"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._current = None
        self._current_raw = None
        self._raw_by_observation.clear()
        self._observed_urls.clear()
        self._loaded_artifacts.clear()

    # --------------------------------------------------------------- request
    def from_request(
        self,
        request: Mapping[str, Any],
        *,
        previous_action: WebAction | None = None,
        previous_observation: Observation | None = None,
    ) -> Observation:
        """Create this step's observation strictly from this request.

        Missing optional new fields simply produce no artifact. A previous
        screenshot, HTML document or DOM tree is never reused.
        """

        if self._closed:
            raise OfficialBackendError("official backend is closed")
        if not isinstance(request, Mapping):
            raise OfficialBackendError("official step request must be a mapping")
        raw = request.get("observation")
        if not isinstance(raw, Mapping):
            raise OfficialBackendError("official request has no observation mapping")
        observation_type = raw.get("type")
        if observation_type is not None and observation_type != "accessibility_tree":
            raise OfficialBackendError(
                f"unsupported official observation type {observation_type!r}; "
                "only accessibility_tree is defined"
            )
        url = raw.get("url")
        if not isinstance(url, str) or not url.strip():
            raise OfficialBackendError("official observation.url is missing")
        text = raw.get("text", "")
        if not isinstance(text, str):
            raise OfficialBackendError("official observation.text must be a string")
        observation_id = f"obs-{self._observation_counter}"
        self._observation_counter += 1
        elements = parse_ax_elements(text, page_id=self.page_id, observation_id=observation_id)
        feedback = None
        previous_action_id = None
        if previous_action is not None:
            if not isinstance(previous_action, WebAction):
                raise OfficialBackendError("previous_action must be a WebAction")
            if previous_action.run_id != self.run_id or previous_action.page_id != self.page_id:
                raise OfficialBackendError("previous_action belongs to another official task/page")
            if (
                previous_observation is not None
                and previous_action.observation_id != previous_observation.observation_id
            ):
                raise OfficialBackendError(
                    "previous_action was not built from the previous official observation"
                )
            previous_action_id = previous_action.action_id
            error_type, error_message = self._platform_error(request, raw)
            if error_type is None:
                status = FeedbackStatus.OBSERVED
                message = f"platform returned a new observation after {previous_action.op.value}"
                error_type = None
            else:
                status = FeedbackStatus.ERROR
                message = error_message or "platform reported an action error"
            page_changed = None
            if previous_observation is not None:
                page_changed = (
                    previous_observation.url != url
                    or previous_observation.ax_text != text
                )
            feedback = Feedback(
                status=status,
                action_id=previous_action.action_id,
                observation_id=observation_id,
                message=message,
                error_type=error_type,
                page_changed=page_changed,
            )
        artifacts, capabilities = self._build_artifacts(raw, observation_id)
        observation = Observation(
            run_id=self.run_id,
            page_id=self.page_id,
            observation_id=observation_id,
            url=url,
            ax_text=text,
            elements=elements,
            previous_action_id=previous_action_id,
            feedback=feedback,
            artifacts=artifacts,
            capabilities=capabilities,
        )
        if url not in self._observed_urls:
            self._observed_urls.append(url)
        self._current = observation
        self._current_raw = raw
        self._raw_by_observation[observation_id] = raw
        if (
            previous_action is not None
            and previous_action.op is ActionOp.TYPE
            and previous_action.submit
        ):
            self._mark_post_submit_observation(previous_action, observation)
        return observation

    # --------------------------------------------------------------- actions
    def validate_action(
        self,
        action: WebAction,
        observation: Observation | None = None,
    ) -> None:
        """Check run/page/observation, AX target and observed-URL navigation."""

        if not isinstance(action, WebAction):
            raise OfficialActionFormatError("official action must be a WebAction")
        current = observation or self._current
        if current is None:
            raise OfficialBackendError("no current official observation")
        if action.run_id != current.run_id:
            raise ActionRejected(
                RejectReason.WRONG_RUN,
                details=["action belongs to another task"],
                correction="Use the action built for the current official observation.",
            )
        if action.page_id != current.page_id:
            raise ActionRejected(
                RejectReason.WRONG_PAGE,
                details=["action belongs to another official page"],
                correction="Use the current official observation page.",
            )
        if action.observation_id != current.observation_id:
            raise ActionRejected(
                RejectReason.STALE_OBSERVATION,
                details=[
                    f"action observation {action.observation_id!r} is not current "
                    f"{current.observation_id!r}"
                ],
                correction="Use only AX numbers from the current observation.",
            )
        # resolve_target applies current capabilities and full binding checks.
        resolve_target(action, current)
        if action.op is ActionOp.GOTO:
            url = action.url or ""
            if url not in self._observed_urls or not same_origin(url, current.url):
                raise ActionRejected(
                    RejectReason.EXTERNAL_ORIGIN,
                    details=[
                        "goto target is not one of this task's observed, same-origin URLs"
                    ],
                    correction=(
                        "goto may only use a same-origin URL already seen in this task; "
                        "the harness will not browse a server-internal URL from this machine."
                    ),
                )
        if action.op in (ActionOp.CLICK, ActionOp.TYPE):
            target = action.target
            if target is None or target not in current.elements:
                raise ActionRejected(
                    RejectReason.UNKNOWN_ELEMENT,
                    details=[f"AX element {target!r} is not in the current observation"],
                    correction="Use an AX number printed in the current observation.",
                )
        if action.op is ActionOp.TYPE and (action.text or "") == "" and not action.submit:
            self._refuse_empty_type(action)

    def _refuse_empty_type(self, action: WebAction) -> None:
        """Refuse an empty non-submitting type locally, with a model-facing reason."""

        error = ActionRejected(
            RejectReason.INVALID_ARGUMENTS,
            details=[
                f"web_type target {action.target!r} has empty text and submit=false; "
                "an empty type has no page effect"
            ],
            correction=EMPTY_TYPE_CORRECTION,
        )
        self._log_empty_type_refusal(action)
        raise error

    def _log_empty_type_refusal(self, action: WebAction) -> None:
        if self.session is None:
            return
        observation = self._current
        try:
            self.session.logger.emit(
                "format_correction",
                page_id=action.page_id,
                url=None if observation is None else observation.url,
                observation_id=action.observation_id,
                action_id=action.action_id,
                action_type=action.op.value,
                result={
                    "reason": "empty_type_text",
                    "scope": "action",
                    "tool": "web_type",
                    "correction": EMPTY_TYPE_CORRECTION,
                    "dispatched": False,
                },
            )
        except Exception:  # noqa: BLE001 - logging must not replace the refusal
            return

    def serialize_action(
        self,
        action: WebAction,
        observation: Observation | None = None,
    ) -> str:
        """Return exactly one official action command."""

        self.validate_action(action, observation)
        op = action.op
        if op is ActionOp.CLICK:
            return f"click [{action.target}]"
        if op is ActionOp.TYPE:
            text = action.text or ""
            if text == "" and action.submit:
                # Equivalent to pressing Enter on the focused field. The platform
                # rejects ``type [n] [] [1]`` and returns no feedback, so that
                # command must never reach the wire.
                return TYPE_SUBMIT_ENTER_COMMAND
            text = _wire_text(text, "type.text", self.text_mode)
            submit = "1" if action.submit else "0"
            return f"type [{action.target}] [{text}] [{submit}]"
        if op is ActionOp.PRESS:
            keys = _wire_text(action.keys or "", "press.keys", self.text_mode)
            if not keys:
                raise OfficialActionFormatError("press.keys must not be empty")
            return f"press [{keys}]"
        if op is ActionOp.SCROLL:
            return f"scroll [{action.direction}]"
        if op is ActionOp.GOTO:
            return f"goto [{_escape_url(action.url or '')}]"
        if op is ActionOp.GO_BACK:
            return "go_back"
        if op is ActionOp.GO_FORWARD:
            return "go_forward"
        raise OfficialActionFormatError(f"unsupported official action {op!r}")

    def serialize_stop(self, message: str) -> str:
        return serialize_stop(message, text_mode=self.text_mode)

    # -------------------------------------------------------------- artifacts
    def _build_artifacts(
        self,
        raw: Mapping[str, Any],
        observation_id: str,
    ) -> tuple[ArtifactSet, Capabilities]:
        screenshot = self._artifact("screenshot", raw.get("screenshot"), raw, observation_id)
        html = self._artifact("html", raw.get("html"), raw, observation_id)
        dom = self._artifact("dom", raw.get("dom"), raw, observation_id)
        capabilities = Capabilities(
            ax_text=True,
            screenshot=screenshot is not None,
            html=html is not None,
            dom=dom is not None,
            actions=frozenset(ActionOp),
        )
        return (
            ArtifactSet(screenshot=screenshot, html=html, dom=dom),
            capabilities,
        )

    def _artifact(
        self,
        kind: str,
        raw_meta: Any,
        raw_observation: Mapping[str, Any],
        observation_id: str,
    ) -> Artifact | None:
        if raw_meta is None:
            return None
        if not isinstance(raw_meta, Mapping):
            return self._failed_artifact(
                kind,
                "invalid_metadata",
                f"{kind} metadata is not a mapping",
                raw_meta if isinstance(raw_meta, Mapping) else {},
                observation_id,
            )
        if raw_meta.get("available") is False:
            reason = str(raw_meta.get("reason") or "unavailable")
            return Artifact(
                kind=kind,
                available=False,
                reason=reason,
                metadata=dict(raw_meta),
            )
        if kind == "dom":
            return self._dom_artifact(raw_meta, observation_id)
        id_key = "image_id" if kind == "screenshot" else "html_id"
        if not isinstance(raw_meta.get(id_key), str):
            return self._failed_artifact(
                kind,
                f"missing_{id_key}",
                f"{kind} metadata has no {id_key}",
                raw_meta,
                observation_id,
            )
        if kind not in self.download_artifacts:
            return Artifact(
                kind=kind,
                available=False,
                reason="not_downloaded",
                metadata=dict(raw_meta),
            )
        try:
            artifact = self._materialize(kind, raw_meta, raw_observation, observation_id)
            self._loaded_artifacts[(observation_id, kind)] = artifact
            return artifact
        except DeadlineExceeded:
            raise
        except OfficialArtifactError as exc:
            return self._failed_artifact(
                kind,
                exc.reason,
                exc.message,
                raw_meta,
                observation_id,
            )
        except Exception as exc:  # noqa: BLE001 - a failed artifact must not break AX
            return self._failed_artifact(
                kind,
                "download_failed",
                f"{kind} download failed: {type(exc).__name__}",
                raw_meta,
                observation_id,
            )

    def _dom_artifact(self, raw_meta: Mapping[str, Any], observation_id: str) -> Artifact:
        root = raw_meta.get("root")
        if not isinstance(root, Mapping):
            return self._failed_artifact(
                "dom",
                "missing_root",
                "dom metadata has no root mapping",
                raw_meta,
                observation_id,
            )
        try:
            encoded = json.dumps(root, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            return self._failed_artifact(
                "dom",
                "invalid_root",
                f"dom root is not JSON-serializable: {type(exc).__name__}",
                raw_meta,
                observation_id,
            )
        if len(encoded) > _DOM_LIMIT:
            return self._failed_artifact(
                "dom",
                "size_limit",
                "dom root exceeds the 4 MiB observation limit",
                raw_meta,
                observation_id,
            )
        if self.evidence is None:
            return self._failed_artifact(
                "dom",
                "no_evidence_store",
                "dom root cannot be saved without an evidence store",
                raw_meta,
                observation_id,
            )
        try:
            return self.evidence.save_bytes(
                "dom",
                encoded,
                observation_id=observation_id,
                suffix=".json",
                metadata={
                    key: value
                    for key, value in raw_meta.items()
                    if key != "root"
                },
            )
        except Exception as exc:  # noqa: BLE001
            return self._failed_artifact(
                "dom",
                "evidence_write_failed",
                f"dom root could not be saved: {type(exc).__name__}",
                raw_meta,
                observation_id,
            )

    def _materialize(
        self,
        kind: str,
        raw_meta: Mapping[str, Any],
        raw_observation: Mapping[str, Any],
        observation_id: str,
    ) -> Artifact:
        if self.evidence is None:
            raise OfficialArtifactError(
                "no_evidence_store", "artifact cannot be stored without an evidence store"
            )
        if kind == "screenshot":
            return self._materialize_screenshot(raw_meta, raw_observation, observation_id)
        if kind == "html":
            return self._materialize_html(raw_meta, observation_id)
        raise OfficialArtifactError("unsupported_artifact", f"cannot download {kind}")

    def _materialize_screenshot(
        self,
        raw_meta: Mapping[str, Any],
        raw_observation: Mapping[str, Any],
        observation_id: str,
    ) -> Artifact:
        self._check_deadline("screenshot download")
        image_id = raw_meta.get("image_id")
        if not isinstance(image_id, str) or not _HEX32_RE.match(image_id):
            raise OfficialArtifactError("invalid_image_id", "screenshot image_id is invalid")
        browser = self._browser()
        data: bytes
        reference: str | None = None
        metadata_complete = self._screenshot_sdk_metadata(raw_meta)
        if metadata_complete and callable(getattr(browser, "save_screenshot", None)):
            try:
                path = browser.save_screenshot(raw_observation, str(self.evidence.screenshots_dir))
            except Exception as exc:  # noqa: BLE001 - SDK verification failure is visible
                raise OfficialArtifactError(
                    "sdk_verification_failed",
                    f"third_party_agent.browser.save_screenshot rejected the PNG: "
                    f"{type(exc).__name__}",
                ) from exc
            if path is None:
                raise OfficialArtifactError(
                    "download_failed", "save_screenshot returned no path"
                )
            resolved = Path(path).resolve()
            screenshots_dir = self.evidence.screenshots_dir.resolve()
            if not resolved.is_relative_to(screenshots_dir):
                raise OfficialArtifactError(
                    "unsafe_path", "save_screenshot wrote outside the evidence screenshot directory"
                )
            try:
                data = resolved.read_bytes()
            except OSError as exc:
                raise OfficialArtifactError(
                    "download_failed", f"cannot read saved screenshot: {type(exc).__name__}"
                ) from exc
            reference = resolved.relative_to(self.evidence.run_dir.resolve()).as_posix()
        else:
            getter = getattr(browser, "get_screenshot", None)
            if not callable(getter):
                raise OfficialArtifactError(
                    "sdk_unavailable", "third_party_agent.browser.get_screenshot is unavailable"
                )
            try:
                data = getter(image_id)
            except Exception as exc:  # noqa: BLE001
                raise OfficialArtifactError(
                    "download_failed",
                    f"screenshot download failed: {type(exc).__name__}",
                ) from exc
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise OfficialArtifactError("download_failed", "screenshot download did not return bytes")
            data = bytes(data)
        self._verify_screenshot(data, raw_meta)
        self._check_deadline("screenshot processing")
        if reference is None:
            artifact = self.evidence.save_bytes(
                "screenshot",
                data,
                observation_id=observation_id,
                metadata=self._artifact_metadata(raw_meta),
            )
        else:
            artifact = Artifact(
                kind="screenshot",
                available=True,
                reference=reference,
                sha256=hashlib.sha256(data).hexdigest(),
                byte_size=len(data),
                metadata=self._artifact_metadata(raw_meta),
            )
        self.evidence.assert_valid(artifact)
        return artifact

    def _materialize_html(self, raw_meta: Mapping[str, Any], observation_id: str) -> Artifact:
        self._check_deadline("HTML download")
        html_id = raw_meta.get("html_id")
        if not isinstance(html_id, str) or not _HEX32_RE.match(html_id):
            raise OfficialArtifactError("invalid_html_id", "HTML html_id is invalid")
        browser = self._browser()
        getter = getattr(browser, "get_html", None)
        if not callable(getter):
            raise OfficialArtifactError(
                "sdk_unavailable", "third_party_agent.browser.get_html is unavailable"
            )
        try:
            text = getter(html_id)
        except Exception as exc:  # noqa: BLE001
            raise OfficialArtifactError(
                "download_failed", f"HTML download failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(text, str):
            raise OfficialArtifactError("download_failed", "HTML download did not return text")
        encoded = text.encode("utf-8")
        expected_size = raw_meta.get("byte_size")
        expected_hash = raw_meta.get("sha256")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            raise OfficialArtifactError("invalid_metadata", "HTML metadata byte_size is invalid")
        if not isinstance(expected_hash, str) or not _HEX64_RE.match(expected_hash):
            raise OfficialArtifactError("invalid_metadata", "HTML metadata sha256 is invalid")
        actual_hash = hashlib.sha256(encoded).hexdigest()
        if len(encoded) != expected_size or actual_hash != expected_hash:
            raise OfficialArtifactError(
                "integrity_mismatch",
                "HTML bytes do not match observation metadata "
                f"(expected {expected_size} bytes/{expected_hash}, "
                f"got {len(encoded)} bytes/{actual_hash})",
            )
        self._check_deadline("HTML processing")
        artifact = self.evidence.save_text(
            "html",
            text,
            observation_id=observation_id,
            metadata=self._artifact_metadata(raw_meta),
        )
        self.evidence.assert_valid(artifact)
        return artifact

    def download_artifact(
        self,
        kind: str,
        observation: Observation | None = None,
    ) -> Artifact:
        """Download one artifact on demand during the current callback.

        Returns an unavailable ``Artifact`` with a concrete reason when the
        metadata is missing/invalid or the bytes fail verification. It never
        substitutes an artifact from another observation.
        """

        if kind not in {"screenshot", "html", "dom"}:
            raise OfficialBackendError(f"unknown artifact kind {kind!r}")
        current = observation or self._current
        if current is None:
            raise OfficialBackendError("no current official observation")
        artifact = current.artifacts.get(kind)
        if artifact is None:
            return Artifact(
                kind=kind,
                available=False,
                reason="not_in_this_observation",
                metadata={},
            )
        if artifact.available:
            return artifact
        cached = self._loaded_artifacts.get((current.observation_id, kind))
        if cached is not None:
            return cached
        if artifact.reason != "not_downloaded":
            return artifact
        raw_observation = self._raw_by_observation.get(current.observation_id)
        if raw_observation is None:
            return self._failed_artifact(
                kind,
                "missing_source_observation",
                "the original observation payload is no longer available",
                artifact.metadata,
                current.observation_id,
            )
        raw_meta = raw_observation.get(kind)
        if not isinstance(raw_meta, Mapping):
            return self._failed_artifact(
                kind,
                "invalid_metadata",
                f"{kind} metadata is missing from the original observation",
                artifact.metadata,
                current.observation_id,
            )
        try:
            loaded = self._materialize(kind, raw_meta, raw_observation, current.observation_id)
            self._loaded_artifacts[(current.observation_id, kind)] = loaded
            return loaded
        except DeadlineExceeded:
            raise
        except OfficialArtifactError as exc:
            return self._failed_artifact(
                kind, exc.reason, exc.message, raw_meta, current.observation_id
            )

    # --------------------------------------------------------------- internals
    def _mark_post_submit_observation(
        self,
        action: WebAction,
        observation: Observation,
    ) -> None:
        """Record that a submitted action produced the new observation.

        This is an event-only forensics marker: the EvidenceStore interface is
        unchanged and the observation is still saved by the Session through its
        normal path. The marker lets a later audit find the observation that must
        be inspected after a save/submit, without treating the page's "updated"
        banner as proof that the submitted value is correct.
        """

        if self.session is None:
            return
        artifacts = {
            kind: {
                "available": artifact.available,
                "reason": artifact.reason,
                "reference": artifact.reference,
            }
            for kind in ("screenshot", "html", "dom")
            if (artifact := observation.artifacts.get(kind)) is not None
        }
        try:
            self.session.logger.emit(
                "artifact_saved",
                page_id=observation.page_id,
                url=observation.url,
                observation_id=observation.observation_id,
                action_id=action.action_id,
                action_type=action.op.value,
                result={
                    "kind": "post_submit_observation",
                    "post_submit_evidence": True,
                    "submit_action_id": action.action_id,
                    "submit_target": action.target,
                    "submit_text_chars": len(action.text or ""),
                    "submit_was_empty": (action.text or "") == "",
                    "observation_id": observation.observation_id,
                    "artifacts": artifacts,
                },
            )
        except Exception:  # noqa: BLE001 - a marker must not break the observation
            return

    def _platform_error(
        self,
        request: Mapping[str, Any],
        raw_observation: Mapping[str, Any],
    ) -> tuple[str | None, str | None]:
        value = (
            request.get("action_error")
            if request.get("action_error") is not None
            else request.get("error", raw_observation.get("error"))
        )
        if value is None:
            return None, None
        if isinstance(value, Mapping):
            error_type = value.get("type") or value.get("error_type") or "platform_action_error"
            message = value.get("message") or value.get("detail") or "platform reported an error"
            return str(error_type), str(message)
        return "platform_action_error", str(value)

    def _screenshot_sdk_metadata(self, raw_meta: Mapping[str, Any]) -> bool:
        mime = raw_meta.get("mime_type")
        width = raw_meta.get("width")
        height = raw_meta.get("height")
        return (
            mime == "image/png"
            and isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and isinstance(height, int)
            and not isinstance(height, bool)
            and height > 0
        )

    def _verify_screenshot(self, data: bytes, raw_meta: Mapping[str, Any]) -> None:
        expected_size = raw_meta.get("byte_size")
        expected_hash = raw_meta.get("sha256")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            raise OfficialArtifactError("invalid_metadata", "screenshot metadata byte_size is invalid")
        if not isinstance(expected_hash, str) or not _HEX64_RE.match(expected_hash):
            raise OfficialArtifactError("invalid_metadata", "screenshot metadata sha256 is invalid")
        actual_hash = hashlib.sha256(data).hexdigest()
        if len(data) != expected_size or actual_hash != expected_hash:
            raise OfficialArtifactError(
                "integrity_mismatch",
                "screenshot bytes do not match observation metadata "
                f"(expected {expected_size} bytes/{expected_hash}, "
                f"got {len(data)} bytes/{actual_hash})",
            )
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
            raise OfficialArtifactError("invalid_png", "screenshot bytes are not a PNG")
        width = raw_meta.get("width")
        height = raw_meta.get("height")
        if isinstance(width, int) and isinstance(height, int):
            actual_width, actual_height = struct.unpack(">II", data[16:24])
            if (actual_width, actual_height) != (width, height):
                raise OfficialArtifactError(
                    "integrity_mismatch",
                    f"screenshot dimensions {actual_width}x{actual_height} do not match "
                    f"metadata {width}x{height}",
                )

    def _artifact_metadata(self, raw_meta: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "image_id",
            "html_id",
            "step",
            "byte_size",
            "sha256",
            "mime_type",
            "width",
            "height",
            "reason",
        }
        metadata = {key: value for key, value in raw_meta.items() if key in allowed}
        metadata["source"] = "official_observation"
        return metadata

    def _failed_artifact(
        self,
        kind: str,
        reason: str,
        message: str,
        raw_meta: Mapping[str, Any] | Any,
        observation_id: str,
    ) -> Artifact:
        metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        metadata["error"] = message
        self.artifact_errors.append(
            {
                "kind": kind,
                "reason": reason,
                "message": message,
                "observation_id": observation_id,
            }
        )
        self._log_artifact_error(kind, reason, message, observation_id)
        return Artifact(
            kind=kind,
            available=False,
            reason=reason,
            metadata=metadata,
        )

    def _log_artifact_error(
        self,
        kind: str,
        reason: str,
        message: str,
        observation_id: str,
    ) -> None:
        if self.session is None:
            return
        observation = self._current
        try:
            self.session.logger.emit(
                "artifact_error",
                page_id=self.page_id,
                url=None if observation is None else observation.url,
                observation_id=observation_id,
                result={
                    "kind": kind,
                    "reason": reason,
                    "message": message,
                },
            )
        except Exception:  # noqa: BLE001 - logging must not break the observation
            return

    def _check_deadline(self, stage: str) -> None:
        if self.deadline_check is not None:
            self.deadline_check(stage)

    def _browser(self) -> Any:
        if self.browser_api is None:
            try:
                from third_party_agent import browser as browser_api  # type: ignore[import-not-found]
            except Exception as exc:  # noqa: BLE001
                raise OfficialArtifactError(
                    "sdk_unavailable",
                    "cannot import third_party_agent.browser; run from code/official_client",
                ) from exc
            self.browser_api = browser_api
        return self.browser_api

__all__ = [
    "DEFAULT_TEXT_MODE",
    "EMPTY_TYPE_CORRECTION",
    "OfficialActionFormatError",
    "OfficialArtifactError",
    "OfficialBackend",
    "OfficialBackendError",
    "TYPE_SUBMIT_ENTER_COMMAND",
    "TextMode",
    "normalize_text_mode",
    "parse_ax_elements",
    "serialize_stop",
    "validate_action_text",
]
