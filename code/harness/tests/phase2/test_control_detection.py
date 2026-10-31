"""E05 visible-state takeover detection: login and verification overlays.

The observations below are plain protocol values with visible text only; no
site store, fault plan or hidden judge state exists in these tests.
"""

from __future__ import annotations

from webagent.control import detect_takeover


LOGIN_AX = """TITLE: Harbor Market — Sign in required
URL: http://fixture.local/product/P-1
HEADINGS:
- h1: Harbor Market & Forum
- h1: Sign in required
TEXT:
- Your session has expired. Sign in in this browser to restore the page.
INTERACTIVE ELEMENTS (1):
[1] role=button name="Sign in"
"""

VERIFICATION_AX = """TITLE: Harbor Market — Verification required
URL: http://fixture.local/product/P-1
HEADINGS:
- h1: Verification required
TEXT:
- The site needs a human verification step before continuing.
INTERACTIVE ELEMENTS (1):
[1] role=button name="Complete verification"
"""

ORDINARY_AX = """TITLE: Harbor Market — Products
URL: http://fixture.local/products
HEADINGS:
- h1: Products
TEXT:
- A generated catalog of everyday objects.
INTERACTIVE ELEMENTS (2):
[1] role=link name="Products"
[2] role=button name="Apply filters"
"""


def test_login_overlay_is_detected_from_visible_heading(make_observation) -> None:
    observation = make_observation(ax_text=LOGIN_AX, elements={"1": ("button", "Sign in")})
    signal = detect_takeover(observation)
    assert signal is not None
    assert signal.kind == "login"
    assert signal.reason == "login_overlay_visible"
    assert "sign in required" in signal.headings
    assert signal.to_dict()["detection_source"] == "visible_observation_only"


def test_verification_overlay_is_detected_from_visible_heading(make_observation) -> None:
    observation = make_observation(
        ax_text=VERIFICATION_AX,
        elements={"1": ("button", "Complete verification")},
    )
    signal = detect_takeover(observation)
    assert signal is not None
    assert signal.kind == "verification"
    assert signal.reason == "verification_overlay_visible"
    assert signal.matched_controls


def test_overlay_copy_with_visible_control_is_enough(make_observation) -> None:
    ax = """TITLE: Session expired
URL: http://fixture.local/post/T-1
TEXT:
- Your session has expired. Sign in to restore the page.
INTERACTIVE ELEMENTS (1):
[1] role=button name="Sign in"
"""
    observation = make_observation(ax_text=ax, elements={"1": ("button", "Sign in")})
    signal = detect_takeover(observation)
    assert signal is not None and signal.kind == "login"


def test_ordinary_page_with_a_sign_in_link_does_not_trigger(make_observation) -> None:
    ax = ORDINARY_AX.replace(
        '[1] role=link name="Products"',
        '[1] role=link name="Sign in"',
    )
    observation = make_observation(ax_text=ax, elements={"1": ("link", "Sign in")})
    assert detect_takeover(observation) is None


def test_no_detection_without_a_current_observation() -> None:
    assert detect_takeover(None) is None
