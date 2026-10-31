/**
 * effGen web surfaces — shared keyboard layer.
 *
 * Loaded by both the dashboard and the playground, this file provides:
 *  - a command palette opened with Cmd/Ctrl-K: type to filter the actions the
 *    host page registers, Up/Down to move, Enter to invoke, Escape to close
 *  - a shortcut reference opened with "?"
 *  - a modal shell with a focus trap and focus restored to the element that
 *    was focused before the overlay opened
 *  - one theme storage key shared by every effGen web surface, reading the
 *    older per-surface keys once so an existing choice carries over
 *
 * Matching, ordering and selection are implemented as a state machine with no
 * DOM access, so the keyboard behavior can be exercised outside a browser; the
 * DOM layer below renders that state. No external asset is referenced.
 */
(function (global) {
  "use strict";

  // ------------------------------------------------------------------
  // Theme preference, shared across surfaces
  // ------------------------------------------------------------------
  var THEME_KEY = "effgen-theme";
  var LEGACY_THEME_KEYS = ["effgen-dashboard-theme", "effgen-playground-theme"];
  var RECENT_KEY = "effgen-recent-actions";
  var MAX_RECENT = 8;

  function storageGet(key) {
    try {
      return global && global.localStorage ? global.localStorage.getItem(key) : null;
    } catch (err) {
      return null; // storage blocked
    }
  }

  function storageSet(key, value) {
    try {
      if (global && global.localStorage) global.localStorage.setItem(key, value);
    } catch (err) { /* storage blocked */ }
  }

  /** The stored theme choice, or null when the viewer has not chosen one. */
  function readStoredTheme() {
    var value = storageGet(THEME_KEY);
    if (value === "dark" || value === "light") return value;
    for (var i = 0; i < LEGACY_THEME_KEYS.length; i++) {
      var legacy = storageGet(LEGACY_THEME_KEYS[i]);
      if (legacy === "dark" || legacy === "light") {
        // Carry the older per-surface choice into the shared key so the next
        // read finds it and both surfaces agree from here on.
        storageSet(THEME_KEY, legacy);
        return legacy;
      }
    }
    return null;
  }

  function storeTheme(theme) {
    if (theme !== "dark" && theme !== "light") return;
    storageSet(THEME_KEY, theme);
  }

  function readRecent() {
    var raw = storageGet(RECENT_KEY);
    if (!raw) return [];
    try {
      var parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.filter(function (id) {
        return typeof id === "string";
      }).slice(0, MAX_RECENT) : [];
    } catch (err) {
      return [];
    }
  }

  function writeRecent(ids) {
    storageSet(RECENT_KEY, JSON.stringify(ids.slice(0, MAX_RECENT)));
  }

  // ------------------------------------------------------------------
  // Matching and ordering (no DOM)
  // ------------------------------------------------------------------
  function haystack(action) {
    return [action.label, action.keywords || "", action.group || "", action.hint || ""]
      .join(" ").toLowerCase();
  }

  /**
   * Relevance of one action for one query. Every whitespace-separated token in
   * the query must appear somewhere in the action's text; a token matching the
   * start of the label or the start of a word in it ranks above one that only
   * matches the keywords. Returns -1 when a token is missing.
   */
  function score(action, query) {
    var q = String(query || "").toLowerCase().trim();
    if (!q) return 0;
    var label = String(action.label || "").toLowerCase();
    var hay = haystack(action);
    var tokens = q.split(/\s+/);
    var total = 0;
    for (var i = 0; i < tokens.length; i++) {
      var token = tokens[i];
      if (hay.indexOf(token) === -1) return -1;
      if (label.indexOf(token) === 0) total += 6;
      else if (label.search(new RegExp("\\b" + token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))) !== -1) total += 4;
      else if (label.indexOf(token) !== -1) total += 3;
      else total += 1;
    }
    return total;
  }

  /**
   * The actions to show for a query, most relevant first. With an empty query
   * the recently-invoked actions lead, followed by the declared order.
   */
  function filterActions(actions, query, recent) {
    var list = (actions || []).slice();
    var recentIds = recent || [];
    var rankOf = function (action) {
      var at = recentIds.indexOf(action.id);
      return at === -1 ? recentIds.length : at;
    };
    var decorated = [];
    for (var i = 0; i < list.length; i++) {
      var value = score(list[i], query);
      if (value < 0) continue;
      decorated.push({ action: list[i], score: value, order: i, recent: rankOf(list[i]) });
    }
    decorated.sort(function (a, b) {
      if (b.score !== a.score) return b.score - a.score;
      if (a.recent !== b.recent) return a.recent - b.recent;
      return a.order - b.order;
    });
    return decorated.map(function (entry) { return entry.action; });
  }

  /**
   * Palette selection as a state machine: the query, the filtered results, and
   * which result is active. `invoke` runs the active action and records it as
   * recently used.
   */
  function createPaletteState(actions, options) {
    var opts = options || {};
    var all = (actions || []).slice();
    var recent = (opts.recent || []).slice();
    var query = "";
    var results = filterActions(all, query, recent);
    var index = results.length ? 0 : -1;

    function recompute() {
      results = filterActions(all, query, recent);
      index = results.length ? 0 : -1;
    }

    return {
      setActions: function (next) { all = (next || []).slice(); recompute(); },
      setQuery: function (value) { query = String(value == null ? "" : value); recompute(); },
      query: function () { return query; },
      results: function () { return results.slice(); },
      count: function () { return results.length; },
      index: function () { return index; },
      /** Move the active row, wrapping at both ends. */
      move: function (delta) {
        if (!results.length) { index = -1; return null; }
        index = ((index + delta) % results.length + results.length) % results.length;
        return results[index];
      },
      first: function () { index = results.length ? 0 : -1; return this.active(); },
      last: function () { index = results.length - 1; return this.active(); },
      active: function () { return index >= 0 && index < results.length ? results[index] : null; },
      setIndex: function (value) {
        if (value >= 0 && value < results.length) index = value;
        return this.active();
      },
      recent: function () { return recent.slice(); },
      invoke: function () {
        var action = this.active();
        if (!action) return null;
        recent = [action.id].concat(recent.filter(function (id) { return id !== action.id; }))
          .slice(0, MAX_RECENT);
        if (opts.onRecent) opts.onRecent(recent.slice());
        if (typeof action.run === "function") action.run();
        return action;
      },
    };
  }

  // ------------------------------------------------------------------
  // DOM layer
  // ------------------------------------------------------------------
  var FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), '
    + 'select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  function isMac() {
    var nav = global && global.navigator;
    var source = (nav && (nav.userAgentData && nav.userAgentData.platform))
      || (nav && nav.platform) || (nav && nav.userAgent) || "";
    return /mac/i.test(String(source));
  }

  /** "⌘K" on Apple platforms, "Ctrl K" elsewhere. */
  function chord(key) {
    return (isMac() ? "⌘" : "Ctrl ") + key;
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function isTypingTarget(el) {
    if (!el) return false;
    var tag = (el.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
  }

  function reducedMotion() {
    try {
      return global.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (err) {
      return false;
    }
  }

  /** Keep Tab inside `container` while it is open. */
  function trapTab(container, event) {
    if (event.key !== "Tab") return;
    var items = Array.prototype.slice.call(container.querySelectorAll(FOCUSABLE))
      .filter(function (el) { return el.offsetParent !== null || el === document.activeElement; });
    if (!items.length) { event.preventDefault(); return; }
    var first = items[0];
    var last = items[items.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  /**
   * Scroll to a section and move real focus to it, so the next Tab continues
   * from there rather than from the top of the page.
   */
  function jumpTo(target) {
    var el = typeof target === "string" ? document.getElementById(target) : target;
    if (!el) return false;
    if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "-1");
    try {
      el.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
    } catch (err) {
      el.scrollIntoView();
    }
    el.focus({ preventScroll: true });
    return true;
  }

  /**
   * Publish the height of the page's sticky chrome as CSS custom properties:
   * `--eff-header-h` positions the section jump row directly under the header
   * instead of on top of it, and `--eff-sticky-h` gives a jumped-to section
   * enough scroll margin to clear both. Re-measured when the window resizes,
   * since the header wraps at narrow widths.
   */
  function trackStickyChrome() {
    var root = document.documentElement;
    function stuckHeight(el) {
      if (!el) return 0;
      var position = "";
      try {
        position = global.getComputedStyle(el).position;
      } catch (err) { /* no layout engine */ }
      return position === "sticky" || position === "fixed" ? el.offsetHeight : 0;
    }
    function measure() {
      var header = stuckHeight(document.querySelector(".header"));
      var nav = stuckHeight(document.querySelector(".eff-panel-nav"));
      root.style.setProperty("--eff-header-h", header + "px");
      root.style.setProperty("--eff-sticky-h", header + nav + "px");
    }
    measure();
    if (global.addEventListener) global.addEventListener("resize", measure);
    return measure;
  }

  /**
   * Speak one short message through a single polite live region.
   *
   * A surface that needs to announce something — a finished race, a re-sorted
   * table — routes it here instead of turning another element into a live
   * region. The element is created once per page and reused.
   *
   * Repeating the message needs care: writing the same string is not a change,
   * and a region that did not change is not read out. So a repeat empties the
   * region and writes the text back on a later tick, which is a change the
   * region does report. A pending repeat is dropped as soon as a different
   * message arrives.
   */
  var announceRepeat = null;
  function announce(message) {
    if (typeof document === "undefined") return null;
    var text = String(message == null ? "" : message);
    var el = document.getElementById("eff-announce");
    if (!el) {
      el = document.createElement("p");
      el.id = "eff-announce";
      el.className = "eff-sr-only";
      el.setAttribute("role", "status");
      el.setAttribute("aria-live", "polite");
      document.body.appendChild(el);
    }
    if (announceRepeat !== null) {
      clearTimeout(announceRepeat);
      announceRepeat = null;
    }
    if (text && el.textContent === text) {
      el.textContent = "";
      announceRepeat = setTimeout(function () {
        announceRepeat = null;
        el.textContent = text;
      }, 60);
      return el;
    }
    el.textContent = text;
    return el;
  }

  function buildOverlay(id, titleText, bodyHtml) {
    var host = document.createElement("div");
    host.className = "eff-overlay";
    host.id = id;
    host.hidden = true;
    host.innerHTML = '<div class="eff-overlay-backdrop" data-close="1"></div>'
      + '<div class="eff-modal" role="dialog" aria-modal="true" aria-labelledby="' + id + '-title">'
      + '<h2 class="eff-sr-only" id="' + id + '-title">' + esc(titleText) + "</h2>"
      + bodyHtml + "</div>";
    document.body.appendChild(host);
    return host;
  }

  function init(config) {
    var cfg = config || {};
    var surface = cfg.surface || "effGen";
    var getActions = cfg.actions || function () { return []; };
    var extraShortcuts = cfg.shortcuts || [];
    var escapeHandlers = [];
    var remeasureChrome = trackStickyChrome();
    // The palette state owns the recently-used order for the rest of the
    // session; it is seeded from storage and written back on every invoke.
    var state = createPaletteState([], {
      recent: readRecent(),
      onRecent: writeRecent,
    });
    var lastFocused = null;

    // ---- palette overlay ----
    var palette = buildOverlay("eff-palette", "Command palette",
      '<div class="eff-palette-head">'
      + '<input id="eff-palette-input" type="text" role="combobox" autocomplete="off"'
      + ' spellcheck="false" aria-expanded="false" aria-controls="eff-palette-list"'
      + ' aria-autocomplete="list" aria-label="Search commands, runs and models"'
      + ' placeholder="Search commands, runs and models…" />'
      + "</div>"
      + '<p class="eff-sr-only" id="eff-palette-status" role="status" aria-live="polite"></p>'
      + '<ul class="eff-palette-list" id="eff-palette-list" role="listbox"'
      + ' aria-label="Commands"></ul>'
      + '<p class="eff-palette-foot">↑↓ move · Enter run · Esc close · ? shortcuts</p>');
    var input = document.getElementById("eff-palette-input");
    var list = document.getElementById("eff-palette-list");
    var status = document.getElementById("eff-palette-status");

    // ---- shortcut reference overlay ----
    var help = buildOverlay("eff-shortcuts", "Keyboard shortcuts",
      '<h3 class="eff-modal-heading">Keyboard shortcuts</h3>'
      + '<dl class="eff-keys" id="eff-keys"></dl>'
      + '<p class="eff-palette-foot">Esc closes this list.</p>');

    function renderHelp() {
      var rows = [
        [chord("K"), "Open the command palette"],
        ["?", "Show this list"],
        ["Esc", "Close the palette, this list, or an open detail"],
        ["↑ ↓", "Move through palette results"],
        ["Enter", "Run the highlighted command"],
      ].concat(extraShortcuts.map(function (s) { return [s.keys, s.label]; }));
      document.getElementById("eff-keys").innerHTML = rows.map(function (row) {
        return "<div><dt><kbd>" + esc(row[0]) + "</kbd></dt><dd>" + esc(row[1]) + "</dd></div>";
      }).join("");
    }

    function renderResults() {
      var results = state.results();
      var activeIndex = state.index();
      if (!results.length) {
        list.innerHTML = '<li class="eff-palette-empty" role="presentation">'
          + "No command matches. Try a model id, a run id, or a panel name.</li>";
      } else {
        var html = "";
        var group = null;
        var open = false;
        results.forEach(function (action, i) {
          if (action.group !== group) {
            if (open) html += "</ul></li>";
            group = action.group;
            html += '<li class="eff-palette-group" role="presentation">'
              + '<span class="eff-palette-group-label">' + esc(group || "Commands") + "</span>"
              + '<ul role="group" aria-label="' + esc(group || "Commands") + '">';
            open = true;
          }
          html += '<li class="eff-palette-option' + (i === activeIndex ? " is-active" : "") + '"'
            + ' id="eff-opt-' + i + '" role="option" data-index="' + i + '"'
            + ' aria-selected="' + (i === activeIndex ? "true" : "false") + '">'
            + '<span class="eff-opt-label">' + esc(action.label) + "</span>"
            + (action.hint ? '<span class="eff-opt-hint">' + esc(action.hint) + "</span>" : "")
            + "</li>";
        });
        if (open) html += "</ul></li>";
        list.innerHTML = html;
      }
      // With no match there is no active option; the attribute is removed
      // rather than emptied, so nothing points at a row that does not exist.
      if (activeIndex >= 0) input.setAttribute("aria-activedescendant", "eff-opt-" + activeIndex);
      else input.removeAttribute("aria-activedescendant");
      // The listbox is only "expanded" while it holds something to choose from.
      input.setAttribute("aria-expanded", results.length ? "true" : "false");
      status.textContent = results.length === 1
        ? "1 command" : results.length + " commands";
      var activeEl = list.querySelector(".eff-palette-option.is-active");
      if (activeEl && activeEl.scrollIntoView) {
        activeEl.scrollIntoView({ block: "nearest" });
      }
    }

    function openPalette() {
      if (!palette.hidden) return;
      closeHelp();
      lastFocused = document.activeElement;
      state.setActions(getActions());
      state.setQuery("");
      input.value = "";
      palette.hidden = false;
      renderResults();
      input.focus();
    }

    function closePalette() {
      if (palette.hidden) return;
      palette.hidden = true;
      if (lastFocused && lastFocused.focus) lastFocused.focus();
      lastFocused = null;
    }

    function openHelp() {
      renderHelp();
      if (!help.hidden) return;
      lastFocused = lastFocused || document.activeElement;
      help.hidden = false;
      var modal = help.querySelector(".eff-modal");
      modal.setAttribute("tabindex", "-1");
      modal.focus();
    }

    function closeHelp() {
      if (help.hidden) return;
      help.hidden = true;
      if (lastFocused && lastFocused.focus) lastFocused.focus();
      lastFocused = null;
    }

    input.addEventListener("input", function () {
      state.setQuery(input.value);
      renderResults();
    });

    input.addEventListener("keydown", function (event) {
      if (event.key === "ArrowDown") { event.preventDefault(); state.move(1); renderResults(); }
      else if (event.key === "ArrowUp") { event.preventDefault(); state.move(-1); renderResults(); }
      else if (event.key === "Home") { event.preventDefault(); state.first(); renderResults(); }
      else if (event.key === "End") { event.preventDefault(); state.last(); renderResults(); }
      else if (event.key === "Enter") {
        event.preventDefault();
        // Close first so focus returns to the page, then let the action move it.
        closePalette();
        state.invoke();
      }
    });

    list.addEventListener("click", function (event) {
      var option = event.target.closest ? event.target.closest(".eff-palette-option") : null;
      if (!option) return;
      state.setIndex(Number(option.dataset.index));
      closePalette();
      state.invoke();
    });

    [palette, help].forEach(function (overlay) {
      overlay.addEventListener("mousedown", function (event) {
        if (event.target && event.target.dataset && event.target.dataset.close) {
          if (overlay === palette) closePalette(); else closeHelp();
        }
      });
      overlay.addEventListener("keydown", function (event) {
        trapTab(overlay.querySelector(".eff-modal"), event);
      });
    });

    document.addEventListener("keydown", function (event) {
      var key = event.key;
      if ((event.metaKey || event.ctrlKey) && (key === "k" || key === "K")) {
        event.preventDefault();
        if (palette.hidden) openPalette(); else closePalette();
        return;
      }
      if (key === "Escape") {
        if (!palette.hidden) { event.preventDefault(); closePalette(); return; }
        if (!help.hidden) { event.preventDefault(); closeHelp(); return; }
        for (var i = 0; i < escapeHandlers.length; i++) {
          if (escapeHandlers[i]() === true) { event.preventDefault(); return; }
        }
        return;
      }
      if (key === "?" && !isTypingTarget(event.target) && palette.hidden) {
        event.preventDefault();
        openHelp();
      }
    });

    return {
      surface: surface,
      open: openPalette,
      close: closePalette,
      openShortcuts: openHelp,
      /** Register a handler run on Escape; return true from it to stop there. */
      onEscape: function (fn) { escapeHandlers.push(fn); },
      state: state,
      jumpTo: jumpTo,
      announce: announce,
      /** Re-measure the sticky chrome after the page changes its header. */
      remeasure: remeasureChrome,
    };
  }

  var api = {
    THEME_KEY: THEME_KEY,
    LEGACY_THEME_KEYS: LEGACY_THEME_KEYS,
    readStoredTheme: readStoredTheme,
    storeTheme: storeTheme,
    readRecent: readRecent,
    score: score,
    filterActions: filterActions,
    createPaletteState: createPaletteState,
    isTypingTarget: isTypingTarget,
    chord: chord,
    jumpTo: jumpTo,
    announce: announce,
    trackStickyChrome: trackStickyChrome,
    init: init,
  };

  if (typeof module === "object" && module.exports) module.exports = api;
  if (global) global.effgenWebUI = api;
})(typeof window !== "undefined" ? window : null);
