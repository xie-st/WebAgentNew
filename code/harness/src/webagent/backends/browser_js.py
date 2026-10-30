"""Shared in-page observation extraction for browser backends.

The JavaScript runs in the page's current rendering context and returns one JSON
string. It is observation only: it reads accessible names, roles, values and a
CSS localization path. It never clicks, types or dispatches business events.

Element numbers are assigned fresh for each call. Playwright and browser-use both
consume this exact extractor, so the model-facing observation has one shape.
"""

from __future__ import annotations

import json
from typing import Any

AX_SNAPSHOT_JS = r"""
() => {
  const MAX_ELEMENTS = 150;
  const MAX_TEXT_LINES = 40;
  const clean = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const truncate = (value, limit) => {
    const text = clean(value);
    return text.length > limit ? text.slice(0, limit - 1) + "…" : text;
  };
  const textOf = (node) => clean(node ? node.textContent : "");
  const attr = (el, name) => clean(el.getAttribute(name));

  const labelText = (el) => {
    const labelledby = el.getAttribute("aria-labelledby");
    if (labelledby) {
      const pieces = labelledby
        .split(/\s+/)
        .map((id) => document.getElementById(id))
        .filter(Boolean)
        .map(textOf);
      if (pieces.length) return pieces.join(" ");
    }
    if (el.id) {
      const explicit = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (explicit) return textOf(explicit);
    }
    const wrapping = el.closest("label");
    if (wrapping) return textOf(wrapping);
    return "";
  };

  const nameFor = (el) => {
    const aria = attr(el, "aria-label");
    if (aria) return truncate(aria, 200);
    const label = labelText(el);
    if (label) return truncate(label, 200);
    const placeholder = attr(el, "placeholder");
    if (placeholder) return truncate(placeholder, 200);
    const alt = attr(el, "alt");
    if (alt) return truncate(alt, 200);
    const title = attr(el, "title");
    if (title) return truncate(title, 200);
    const type = (el.getAttribute("type") || "").toLowerCase();
    if (el.tagName === "INPUT" && ["button", "submit", "reset"].includes(type)) {
      const value = attr(el, "value");
      if (value) return truncate(value, 200);
    }
    return truncate(textOf(el), 200);
  };

  const roleFor = (el) => {
    const explicit = attr(el, "role");
    if (explicit && explicit !== "presentation" && explicit !== "none") return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "button") return "button";
    if (tag === "textarea") return "textbox";
    if (tag === "select") return "combobox";
    if (tag === "summary") return "button";
    if (tag === "input") {
      const type = (el.getAttribute("type") || "text").toLowerCase();
      if (type === "checkbox") return "checkbox";
      if (type === "radio") return "radio";
      if (type === "range") return "slider";
      if (["button", "submit", "reset"].includes(type)) return "button";
      if (type === "file") return "file";
      return "textbox";
    }
    if (el.isContentEditable) return "textbox";
    return "generic";
  };

  const cssPath = (el) => {
    if (!(el instanceof Element)) return "";
    const id = el.getAttribute("id");
    if (id) {
      const escaped = CSS.escape(id);
      try {
        if (document.querySelectorAll("#" + escaped).length === 1) return "#" + escaped;
      } catch (err) {
        // fall through to a structural path
      }
    }
    const parts = [];
    let node = el;
    while (node && node.nodeType === Node.ELEMENT_NODE && node !== document.documentElement) {
      const tag = node.tagName.toLowerCase();
      const parent = node.parentElement;
      if (!parent) break;
      const sameTag = Array.from(parent.children).filter((child) => child.tagName === node.tagName);
      const suffix = sameTag.length > 1 ? `:nth-of-type(${sameTag.indexOf(node) + 1})` : "";
      parts.unshift(tag + suffix);
      node = parent;
      if (node === document.body) break;
    }
    return "body > " + parts.join(" > ");
  };

  const checkVisibility = (el) => {
    if (typeof el.checkVisibility === "function") {
      try {
        if (!el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) return false;
      } catch (err) {
        // fall back to the manual checks below
      }
    }
    const style = getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden" || style.visibility === "collapse") return false;
    if (Number(style.opacity) <= 0) return false;
    if (style.pointerEvents === "none") return false;
    const rect = el.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return false;
    if (el.getAttribute("aria-hidden") === "true" && !el.hasAttribute("tabindex")) return false;
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const inViewport = centerX >= 0 && centerY >= 0 && centerX <= innerWidth && centerY <= innerHeight;
    if (inViewport) {
      const hit = document.elementFromPoint(centerX, centerY);
      if (!hit) return false;
      if (hit !== el && !el.contains(hit) && !hit.contains(el)) return false;
    }
    return true;
  };

  const valueFor = (el) => {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute("type") || "").toLowerCase();
    if (tag === "input" && (type === "checkbox" || type === "radio")) {
      return el.checked ? "checked" : "unchecked";
    }
    if (tag === "input" || tag === "textarea" || tag === "select") {
      if (type === "password") return el.value ? "[value present]" : "";
      if (tag === "select") {
        return Array.from(el.selectedOptions || []).map((option) => clean(option.textContent)).join(", ");
      }
      return truncate(el.value, 300);
    }
    if (el.isContentEditable) return truncate(textOf(el), 300);
    return "";
  };

  const stateFor = (el) => {
    const states = [];
    if (el.disabled) states.push("disabled");
    if (el.getAttribute("aria-disabled") === "true") states.push("aria-disabled");
    if (el.readOnly) states.push("readonly");
    if (el.required) states.push("required");
    if (el.getAttribute("aria-expanded") !== null) states.push("expanded=" + el.getAttribute("aria-expanded"));
    if (el.getAttribute("aria-checked") !== null) states.push("checked=" + el.getAttribute("aria-checked"));
    return states.join(" ");
  };

  const selector = [
    "a[href]", "button", "input:not([type='hidden'])", "textarea", "select",
    "[contenteditable='true']", "[role='button']", "[role='link']", "[role='textbox']",
    "[role='checkbox']", "[role='radio']", "[role='combobox']", "[role='tab']",
    "[role='menuitem']", "[tabindex]:not([tabindex='-1'])"
  ].join(",");

  const seen = new Set();
  const elements = [];
  for (const el of document.querySelectorAll(selector)) {
    if (seen.has(el)) continue;
    seen.add(el);
    const type = (el.getAttribute("type") || "").toLowerCase();
    if (el.tagName === "INPUT" && type === "hidden") continue;
    if (!checkVisibility(el)) continue;
    if (elements.length >= MAX_ELEMENTS) break;
    const role = roleFor(el);
    const name = nameFor(el);
    const value = valueFor(el);
    const href = el.tagName === "A" ? el.href : "";
    const path = cssPath(el);
    if (!path) continue;
    elements.push({
      role,
      name,
      value,
      state: stateFor(el),
      tag: el.tagName.toLowerCase(),
      type,
      href,
      css_path: path
    });
  }

  const headings = [];
  for (const el of document.querySelectorAll("h1, h2, h3")) {
    const text = truncate(textOf(el), 240);
    if (text) headings.push(`${el.tagName.toLowerCase()}: ${text}`);
  }

  const textBlocks = [];
  const root = document.querySelector("main") || document.body;
  for (const el of root.querySelectorAll(
    "p, li, dd, dt, td, th, label, span, [role='status'], [aria-live], div"
  )) {
    if (el.tagName === "DIV" && el.children.length > 0) continue;
    if (el.children.length > 2) continue;
    const text = truncate(textOf(el), 240);
    if (!text || textBlocks.includes(text)) continue;
    textBlocks.push(text);
    if (textBlocks.length >= MAX_TEXT_LINES) break;
  }

  const lines = [];
  lines.push(`TITLE: ${clean(document.title)}`);
  lines.push(`URL: ${location.href}`);
  lines.push(`VIEWPORT: ${innerWidth}x${innerHeight} scrollX=${Math.round(scrollX)} scrollY=${Math.round(scrollY)}`);
  if (headings.length) {
    lines.push("HEADINGS:");
    for (const heading of headings) lines.push(`- ${heading}`);
  }
  if (textBlocks.length) {
    lines.push("TEXT:");
    for (const block of textBlocks) lines.push(`- ${block}`);
  }
  lines.push(`INTERACTIVE ELEMENTS (${elements.length}):`);
  elements.forEach((element, index) => {
    const number = String(index + 1);
    const bits = [`[${number}] role=${element.role}`, `name=${JSON.stringify(element.name)}`];
    if (element.value) bits.push(`value=${JSON.stringify(element.value)}`);
    if (element.state) bits.push(`state=${JSON.stringify(element.state)}`);
    if (element.href) bits.push(`href=${JSON.stringify(element.href)}`);
    lines.push(bits.join(" "));
  });
  return JSON.stringify({
    title: clean(document.title),
    url: location.href,
    viewport: {
      width: innerWidth,
      height: innerHeight,
      scrollX: Math.round(scrollX),
      scrollY: Math.round(scrollY)
    },
    ax_text: lines.join("\n"),
    elements
  });
}
"""


def parse_snapshot(raw: Any) -> dict[str, Any]:
    """Normalize the extractor's JSON string from either browser API."""

    if isinstance(raw, str):
        data = json.loads(raw)
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError("browser observation must be a JSON string or object")
    if not isinstance(data, dict):
        raise ValueError("browser observation is not an object")
    for key in ("title", "url", "ax_text"):
        if not isinstance(data.get(key), str):
            raise ValueError(f"browser observation field {key!r} is not a string")
    if not isinstance(data.get("elements"), list):
        raise ValueError("browser observation elements is not a list")
    return data


__all__ = ["AX_SNAPSHOT_JS", "parse_snapshot"]
