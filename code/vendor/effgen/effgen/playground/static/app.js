/**
 * effGen Playground — frontend app
 *
 * Drives the existing POST /v1/chat/completions endpoint:
 *  - fetches /playground/bootstrap for presets, tool options, defaults, and (in
 *    local-view mode) a session key
 *  - fetches /v1/models/catalog to populate the model picker with real ids,
 *    pricing, and local/free flags
 *  - runs a prompt (streamed or complete), rendering the answer, token/cost
 *    stats, and the tool step trace
 *  - generates copy-as-curl / CLI / Python snippets from the form state
 *
 * Keyboard navigation — the Cmd/Ctrl-K command palette and the "?" shortcut
 * list — comes from webui.js, which the dashboard loads as well. This file
 * supplies the playground's own palette commands.
 *
 * All assets are served locally; the page has no external network dependency.
 * The API key, when the user pastes one, is held in memory for the tab only and
 * is never written to disk.
 */
(function () {
  "use strict";

  // The keyboard layer shared with the dashboard. Absent only if webui.js
  // failed to load, in which case the page keeps working without shortcuts.
  const webui = window.effgenWebUI || null;

  // In-memory config/state (no persistence of the key to disk).
  let apiKey = "";
  let bootstrap = { presets: [], tools: [], defaults: {} };
  let catalog = { data: [], local: [], providers: [] };
  let presetsByName = {};
  let snippetKind = "curl";
  let lastRun = null; // {model, prompt, tools, temperature, maxTokens, system}

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ------------------------------------------------------------------
  // Theme (mirrors the dashboard: OS scheme by default, toggle overrides)
  // ------------------------------------------------------------------
  function currentTheme() {
    const attr = document.documentElement.getAttribute("data-theme");
    if (attr === "dark" || attr === "light") return attr;
    const prefersLight = window.matchMedia
      && window.matchMedia("(prefers-color-scheme: light)").matches;
    return prefersLight ? "light" : "dark";
  }
  function syncThemeButton(announce) {
    const theme = currentTheme();
    const icon = $("theme-icon");
    if (icon) icon.textContent = theme === "dark" ? "☾" : "☀";
    const btn = $("theme-btn");
    if (btn) btn.setAttribute("aria-label",
      theme === "dark" ? "Dark theme active. Switch to light theme."
                       : "Light theme active. Switch to dark theme.");
    const live = $("theme-status");
    // A change the viewer made is announced; the initial sync stays silent.
    if (announce && live) live.textContent = theme === "dark" ? "Dark theme" : "Light theme";
  }
  function applyTheme(theme, persist) {
    document.documentElement.setAttribute("data-theme", theme);
    if (persist && webui) webui.storeTheme(theme);
    syncThemeButton(persist);
  }
  function initTheme() {
    // One preference key is shared by every effGen web surface, so a choice
    // made here also applies on the dashboard.
    const stored = webui ? webui.readStoredTheme() : null;
    if (stored === "dark" || stored === "light") applyTheme(stored, false);
    else syncThemeButton();
    const btn = $("theme-btn");
    if (btn) btn.addEventListener("click", () =>
      applyTheme(currentTheme() === "dark" ? "light" : "dark", true));
  }

  // ------------------------------------------------------------------
  // Banners
  // ------------------------------------------------------------------
  function showAuthBanner(msg) {
    const el = $("auth-banner");
    if (!el) return;
    el.textContent = msg || "This run needs an API key. Paste it above, or restart the "
      + "server with EFFGEN_PUBLIC_PLAYGROUND=1 for local viewing.";
    el.hidden = false;
  }
  function hideAuthBanner() { const el = $("auth-banner"); if (el) el.hidden = true; }
  function showError(msg) {
    const el = $("error-banner");
    if (!el) return;
    el.textContent = msg;
    el.hidden = false;
  }
  function hideError() { const el = $("error-banner"); if (el) el.hidden = true; }

  // ------------------------------------------------------------------
  // Formatting
  // ------------------------------------------------------------------
  function fmtCost(v) {
    if (v == null) return "unpriced";
    if (v === 0) return "$0.00";
    if (v < 0.01) return "$" + v.toFixed(6);
    return "$" + v.toFixed(4);
  }
  function fmtPrice(rec) {
    if (rec.free_tier) return "free";
    if (rec.price_in_per_1m == null && rec.price_out_per_1m == null) return "unpriced";
    const i = rec.price_in_per_1m == null ? "?" : "$" + rec.price_in_per_1m;
    const o = rec.price_out_per_1m == null ? "?" : "$" + rec.price_out_per_1m;
    return i + " / " + o + " per 1M";
  }

  function authHeaders() {
    return apiKey ? { Authorization: "Bearer " + apiKey } : {};
  }

  // ------------------------------------------------------------------
  // Bootstrap + catalog
  // ------------------------------------------------------------------
  async function loadBootstrap() {
    try {
      const resp = await fetch("/playground/bootstrap", { cache: "no-store" });
      if (resp.status === 401) { showAuthBanner(); revealKeyField(); return; }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      bootstrap = await resp.json();
    } catch (err) {
      // Bootstrap is behind auth by default; the user pastes a key and we retry.
      revealKeyField();
      return;
    }
    // A local-view session key means Run works without a paste.
    if (bootstrap.session_key) { apiKey = bootstrap.session_key; hideAuthBanner(); }
    else { revealKeyField(); }

    applyDefaults();
    populatePresets();
    populateTools();
  }

  function revealKeyField() {
    const f = $("key-field");
    if (f) f.hidden = false;
    const input = $("api-key");
    if (input && !input.dataset.wired) {
      input.dataset.wired = "1";
      input.addEventListener("input", () => {
        apiKey = input.value.trim();
        if (apiKey) { hideAuthBanner(); loadCatalog(); }
      });
    }
  }

  function applyDefaults() {
    const d = bootstrap.defaults || {};
    if (d.temperature != null) $("temperature").value = d.temperature;
    if (d.max_tokens != null) $("max-tokens").value = d.max_tokens;
    if (d.stream != null) $("stream-toggle").checked = !!d.stream;
  }

  function populatePresets() {
    const sel = $("preset-select");
    (bootstrap.presets || []).forEach((p) => {
      presetsByName[p.name] = p;
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name + " — " + (p.description || "").slice(0, 60);
      sel.appendChild(opt);
    });
    sel.addEventListener("change", onPresetChange);
  }

  function onPresetChange() {
    const name = $("preset-select").value;
    const hint = $("preset-hint");
    if (!name || !presetsByName[name]) {
      if (hint) hint.textContent = "";
      syncToolChecks([]);
      return;
    }
    const p = presetsByName[name];
    // Apply the preset client-side: its system prompt + tool selection + temp.
    if (p.temperature != null) $("temperature").value = p.temperature;
    syncToolChecks(p.tools || []);
    if (hint) {
      const known = (p.tools || []).filter((t) => (bootstrap.tools || []).includes(t));
      const extra = (p.tools || []).filter((t) => !(bootstrap.tools || []).includes(t));
      let msg = "System prompt applied.";
      if (known.length) msg += " Tools: " + known.join(", ") + ".";
      if (extra.length) msg += " (" + extra.join(", ") + " run server-side if hosted.)";
      hint.textContent = msg;
    }
  }

  function populateTools() {
    const host = $("tool-list");
    host.innerHTML = "";
    (bootstrap.tools || []).forEach((name) => {
      const id = "tool-" + name;
      const label = document.createElement("label");
      label.className = "tool-chip";
      label.innerHTML = `<input type="checkbox" id="${id}" value="${esc(name)}" /> ${esc(name)}`;
      host.appendChild(label);
    });
  }

  function syncToolChecks(names) {
    const set = new Set(names || []);
    (bootstrap.tools || []).forEach((name) => {
      const cb = $("tool-" + name);
      if (cb) cb.checked = set.has(name);
    });
  }

  function selectedTools() {
    // Union of the checkbox tools and any preset-only tools (hosted server-side).
    const checked = (bootstrap.tools || [])
      .filter((name) => { const cb = $("tool-" + name); return cb && cb.checked; });
    const presetName = $("preset-select").value;
    const presetTools = presetName && presetsByName[presetName]
      ? (presetsByName[presetName].tools || []) : [];
    const out = new Set(checked);
    presetTools.forEach((t) => out.add(t));
    return Array.from(out);
  }

  async function loadCatalog() {
    const sel = $("model-select");
    try {
      const resp = await fetch("/v1/models/catalog", {
        cache: "no-store", headers: authHeaders(),
      });
      if (resp.status === 401) { showAuthBanner(); return; }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      catalog = await resp.json();
    } catch (err) {
      sel.innerHTML = '<option value="">Catalog unavailable — type a model id below</option>';
      makeModelFreeText();
      return;
    }
    renderModelOptions();
  }

  // Model ids that name an embedding model. A catalog entry carries no "can
  // this answer a prompt" flag, so the id is what this reads: the names below
  // are the ones that ship as sentence/embedding encoders. This chat surface
  // leaves them out; the dashboard's catalog browser still lists every model.
  const EMBEDDING_ID_SIGNALS = [
    "embedding", "embed-", "-embed", "sentence-transformers/",
    "bge-", "/bge", "minilm", "gte-", "e5-",
  ];

  function isEmbeddingModelId(id) {
    const lowered = String(id || "").toLowerCase();
    return EMBEDDING_ID_SIGNALS.some((signal) => lowered.includes(signal));
  }

  function renderModelOptions() {
    const sel = $("model-select");
    sel.innerHTML = "";

    // Locally-cached models first — the zero-config, no-key-needed entry.
    const locals = (catalog.local || [])
      .filter((m) => m.complete !== false)
      .filter((m) => !isEmbeddingModelId(m.id));
    if (locals.length) {
      const grp = document.createElement("optgroup");
      grp.label = "Local (no cloud key)";
      locals.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.id + "  ·  local";
        opt.dataset.local = "1";
        grp.appendChild(opt);
      });
      sel.appendChild(grp);
    }

    const byProvider = {};
    (catalog.data || []).forEach((rec) => {
      if (rec.deprecated) return;
      if (isEmbeddingModelId(rec.id)) return;
      (byProvider[rec.provider] = byProvider[rec.provider] || []).push(rec);
    });
    Object.keys(byProvider).sort().forEach((prov) => {
      const grp = document.createElement("optgroup");
      grp.label = prov;
      byProvider[prov].forEach((rec) => {
        const opt = document.createElement("option");
        opt.value = rec.provider + ":" + rec.id;
        const tag = rec.free_tier ? "free" : (rec.is_priced ? fmtPrice(rec) : "unpriced");
        opt.textContent = rec.id + "  ·  " + tag;
        opt.dataset.rec = JSON.stringify(rec);
        grp.appendChild(opt);
      });
      sel.appendChild(grp);
    });

    // Default the picker to the server's configured default when present.
    const def = bootstrap.default_model;
    if (def) {
      for (const o of sel.options) {
        if (o.value === def || o.value.endsWith(":" + def) || o.value === "openai:" + def) {
          o.selected = true; break;
        }
      }
    }
    sel.addEventListener("change", onModelChange);
    onModelChange();
  }

  function makeModelFreeText() {
    // Fallback when the catalog can't be read: let the user type any id.
    const sel = $("model-select");
    sel.insertAdjacentHTML("beforeend",
      '<option value="' + esc(bootstrap.default_model || "") + '">'
      + esc(bootstrap.default_model || "effgen-default") + "</option>");
  }

  function onModelChange() {
    const opt = $("model-select").selectedOptions[0];
    const hint = $("model-hint");
    if (!opt || !hint) return;
    if (opt.dataset.local) { hint.textContent = "Local model — runs on this server, unpriced."; return; }
    if (!opt.dataset.rec) { hint.textContent = ""; return; }
    try {
      const rec = JSON.parse(opt.dataset.rec);
      const bits = [fmtPrice(rec)];
      if (rec.context_window) bits.push((rec.context_window / 1000).toFixed(0) + "K ctx");
      if (rec.supports_tools) bits.push("tools");
      if (rec.supports_vision) bits.push("vision");
      if (rec.verified_on) bits.push("verified " + rec.verified_on);
      hint.textContent = bits.join("  ·  ");
    } catch { hint.textContent = ""; }
  }

  // ------------------------------------------------------------------
  // Run
  // ------------------------------------------------------------------
  function buildMessages(system, prompt) {
    const msgs = [];
    if (system && system.trim()) msgs.push({ role: "system", content: system });
    msgs.push({ role: "user", content: prompt });
    return msgs;
  }

  function currentSystemPrompt() {
    const name = $("preset-select").value;
    return name && presetsByName[name] ? (presetsByName[name].system_prompt || "") : "";
  }

  function toolSpecs(names) {
    return names.map((n) => ({ type: "function", function: { name: n } }));
  }

  async function run() {
    hideError();
    const prompt = $("prompt").value.trim();
    if (!prompt) { showError("Enter a prompt first."); return; }
    const model = $("model-select").value || bootstrap.default_model || "effgen-default";
    if (!apiKey && !bootstrap.dev_mode) { showAuthBanner(); revealKeyField(); return; }

    const tools = selectedTools();
    const temperature = parseFloat($("temperature").value);
    const maxTokens = parseInt($("max-tokens").value, 10);
    const stream = $("stream-toggle").checked;
    const system = currentSystemPrompt();

    lastRun = { model, prompt, tools, temperature, maxTokens, system };
    renderSnippet();
    $("snippets").hidden = false;

    const body = {
      model,
      messages: buildMessages(system, prompt),
      temperature: isNaN(temperature) ? undefined : temperature,
      max_tokens: isNaN(maxTokens) ? undefined : maxTokens,
      stream,
    };
    if (tools.length) body.tools = toolSpecs(tools);
    if (stream) body.stream_options = { include_usage: true };

    const btn = $("run-btn");
    btn.disabled = true;
    btn.textContent = "Running…";
    const answerEl = $("answer");
    answerEl.className = "answer";
    answerEl.textContent = "";
    $("stats").hidden = true;
    $("trace-wrap").hidden = true;

    const started = performance.now();
    try {
      if (stream) await runStreaming(body, answerEl);
      else await runComplete(body, answerEl);
    } catch (err) {
      showError(String(err && err.message ? err.message : err));
      answerEl.className = "answer placeholder";
      answerEl.textContent = "Run failed.";
    } finally {
      // Whatever happened, the answer box is no longer receiving deltas.
      answerEl.setAttribute("aria-busy", "false");
      const ms = performance.now() - started;
      const lat = $("stat-latency");
      if (lat && lat.textContent === "—") lat.textContent = (ms / 1000).toFixed(2) + "s";
      btn.disabled = false;
      btn.textContent = "Run";
    }
  }

  async function runComplete(body, answerEl) {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
      body: JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      if (resp.status === 401) { showAuthBanner(); revealKeyField(); }
      const emsg = data && data.error ? data.error.message : ("HTTP " + resp.status);
      throw new Error(emsg);
    }
    const choice = (data.choices || [])[0] || {};
    answerEl.textContent = (choice.message && choice.message.content) || "(no content)";
    renderStats(data);
    renderTrace(data.effgen || {}, body.tools);
  }

  async function runStreaming(body, answerEl) {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 401) { showAuthBanner(); revealKeyField(); }
      throw new Error(data && data.error ? data.error.message : ("HTTP " + resp.status));
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    let usage = null;
    let model = body.model;
    // The answer box is a polite live region. Rewriting it per delta made a
    // screen reader re-read the whole answer from the start on every token, so
    // the deltas are appended as text nodes instead and the region is marked
    // busy for the duration: one announcement at the end, not one per token.
    // The cursor lives outside the announced text.
    answerEl.textContent = "";
    answerEl.setAttribute("aria-busy", "true");
    const answerText = document.createTextNode("");
    answerEl.appendChild(answerText);
    const cursorEl = document.createElement("span");
    cursorEl.className = "cursor";
    cursorEl.setAttribute("aria-hidden", "true");
    cursorEl.textContent = "▋";
    answerEl.appendChild(cursorEl);
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const evt of events) {
        const line = evt.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let obj;
        try { obj = JSON.parse(payload); } catch { continue; }
        if (obj.error) throw new Error(obj.error.message || "stream error");
        if (obj.model) model = obj.model;
        if (obj.usage) usage = obj.usage;
        const delta = (obj.choices && obj.choices[0] && obj.choices[0].delta) || {};
        if (delta.content) {
          text += delta.content;
          answerText.appendData(delta.content);
          answerEl.scrollTop = answerEl.scrollHeight;
        }
      }
    }
    answerEl.textContent = text || "(no content)";
    answerEl.setAttribute("aria-busy", "false");
    renderStats({ model, usage, effgen: {} });
    // The streamed path returns tokens, not the tool trace; a non-streamed run
    // surfaces the step trace.
    if (body.tools && body.tools.length) {
      $("trace-wrap").hidden = false;
      $("trace").innerHTML = '<p class="trace-empty">Turn off streaming to see the tool step trace.</p>';
    }
  }

  function renderStats(data) {
    const u = data.usage || {};
    const eff = data.effgen || {};
    $("stat-model").textContent = data.model || eff.resolved_model || "—";
    const pin = u.prompt_tokens == null ? "?" : u.prompt_tokens;
    const pout = u.completion_tokens == null ? "?" : u.completion_tokens;
    $("stat-tokens").textContent = pin + " / " + pout;
    $("stat-total").textContent = u.total_tokens == null ? "—" : u.total_tokens;
    $("stat-cost").textContent = fmtCost(eff.cost_usd);
    $("stat-latency").textContent = "—";
    $("stats").hidden = false;
  }

  function renderTrace(eff, toolsSent) {
    const wrap = $("trace-wrap");
    const host = $("trace");
    const steps = eff.trace || [];
    if (!steps.length) {
      if (toolsSent && toolsSent.length) {
        wrap.hidden = false;
        host.innerHTML = '<p class="trace-empty">No tool was invoked for this prompt.</p>';
      } else {
        wrap.hidden = true;
      }
      return;
    }
    wrap.hidden = false;
    host.innerHTML = steps.map((s) => {
      const cls = s.ok === false ? "err" : "ok";
      const dur = s.duration_ms != null ? s.duration_ms.toFixed(0) + "ms" : "";
      return `<div class="trace-step ${cls}">`
        + `<span class="trace-tool">${esc(s.tool)}</span>`
        + `<span class="trace-args">${esc(s.args || "")}</span>`
        + `<span class="trace-arrow">→</span>`
        + `<span class="trace-result">${esc(s.result_summary || "")}</span>`
        + `<span class="trace-dur">${dur}</span></div>`;
    }).join("");
  }

  // ------------------------------------------------------------------
  // Battle — several models answering one prompt at once
  //
  // Each contender is one streamed POST /v1/chat/completions with
  // stream_options.include_usage, so a battle adds no new model-execution path
  // and inherits the endpoint's auth and spend controls. The per-run cost comes
  // from the effgen extension on the usage chunk, which keeps one source of
  // truth for pricing instead of re-deriving it here.
  // ------------------------------------------------------------------
  function currentMode() {
    const battle = $("mode-battle");
    return battle && battle.checked ? "battle" : "single";
  }

  function syncMode() {
    const battle = currentMode() === "battle";
    $("model-field").hidden = battle;
    $("battle-field").hidden = !battle;
    $("single-results").hidden = battle;
    $("battle-results").hidden = !battle;
    const btn = $("run-btn");
    if (btn) btn.textContent = battle ? "Start battle" : "Run";
  }

  function populateBattleModels() {
    // The same catalog entries the single-run picker uses, so contenders carry
    // real ids, prices and local flags rather than served aliases.
    const src = $("model-select");
    const dst = $("battle-models");
    if (!src || !dst) return;
    dst.innerHTML = "";
    const groups = src.querySelectorAll("optgroup");
    if (!groups.length) {
      // The catalog could not be read; offer whatever the picker fell back to.
      for (const opt of src.querySelectorAll("option")) {
        if (!opt.value) continue;
        const clone = document.createElement("option");
        clone.value = opt.value;
        clone.textContent = opt.textContent;
        dst.appendChild(clone);
      }
      return;
    }
    for (const group of groups) {
      const grp = document.createElement("optgroup");
      grp.label = group.label;
      for (const opt of group.querySelectorAll("option")) {
        const clone = document.createElement("option");
        clone.value = opt.value;
        clone.textContent = opt.textContent;
        grp.appendChild(clone);
      }
      dst.appendChild(grp);
    }
  }

  function selectedContenders() {
    const sel = $("battle-models");
    if (!sel) return [];
    return Array.from(sel.selectedOptions).map((o) => o.value).filter(Boolean);
  }

  function renderBattleColumns(entries) {
    const host = $("battle-grid");
    if (!host) return;
    // Each column is a named group with its own heading, so heading and group
    // navigation move between contenders instead of reading straight through.
    host.innerHTML = entries.map((e, i) => (
      `<div class="battle-col" data-i="${i}" role="group" aria-labelledby="battle-name-${i}">`
      + `<div class="battle-head"><h3 class="battle-model" id="battle-name-${i}">${esc(e.model)}</h3>`
      + `<span class="battle-state" data-state="${esc(e.state)}">${esc(stateLabel(e))}</span></div>`
      + `<div class="battle-answer">${e.error ? '<span class="battle-err">' + esc(e.error) + "</span>" : esc(e.text)}</div>`
      + `<div class="battle-foot">${esc(battleFooter(e))}</div>`
      + "</div>"
    )).join("");
  }

  // One polite live region for everything this page announces; the shared
  // keyboard layer owns it. Without webui.js the page still works, silently.
  function announce(message) {
    if (webui && webui.announce) webui.announce(message);
  }

  function stateLabel(e) {
    if (e.state === "waiting") return "waiting for first token…";
    if (e.state === "streaming") return "generating…";
    if (e.state === "failed") return "failed";
    if (e.state === "done") return "done";
    return "queued";
  }

  function battleFooter(e) {
    const bits = [];
    if (e.ttft != null) bits.push("ttft " + (e.ttft / 1000).toFixed(2) + "s");
    if (e.latency != null) bits.push("total " + (e.latency / 1000).toFixed(2) + "s");
    if (e.usage && e.usage.total_tokens != null) bits.push(e.usage.total_tokens + " tok");
    if (e.state === "done") bits.push(fmtCost(e.cost));
    return bits.join("  ·  ") || stateLabel(e);
  }

  function renderBattleVerdict(entries, wallMs) {
    const host = $("battle-verdict");
    if (!host) return;
    // A contender that failed never wins: it spent nothing and took no time,
    // which would make it the trivial winner on both measures.
    const finishers = entries.filter((e) => !e.error && e.text);
    const rows = [];
    const timed = finishers.filter((e) => e.latency != null);
    if (timed.length) {
      const fastest = timed.reduce((a, b) => (b.latency < a.latency ? b : a));
      rows.push(["Fastest", esc(fastest.model) + " — answered in "
        + (fastest.latency / 1000).toFixed(2) + "s"]);
    }
    const priced = finishers.filter((e) => e.cost != null);
    if (priced.length) {
      const cheapest = priced.reduce((a, b) => (b.cost < a.cost ? b : a));
      const unpriced = finishers.length - priced.length;
      rows.push(["Cheapest", esc(cheapest.model) + " — " + fmtCost(cheapest.cost)
        + " for this run"
        + (unpriced ? "; " + unpriced + " model(s) publish no price and were not ranked" : "")]);
    }
    if (finishers.length) {
      const longest = finishers.reduce((a, b) => (b.text.length > a.text.length ? b : a));
      rows.push(["Longest", esc(longest.model) + " — " + longest.text.length + " characters"]);
    }
    const failed = entries.filter((e) => e.error);
    if (failed.length) {
      rows.push(["Did not answer",
        failed.map((e) => esc(e.model) + " (" + esc(e.error) + ")").join(", ")]);
    }
    const total = priced.length ? priced.reduce((s, e) => s + e.cost, 0) : null;
    rows.push(["Race", finishers.length + "/" + entries.length + " answered in "
      + (wallMs / 1000).toFixed(2) + "s  ·  total " + fmtCost(total)]);

    host.innerHTML = "<h3>Verdict</h3>" + rows.map(
      ([k, v]) => `<div class="verdict-row"><span class="verdict-key">${esc(k)}</span>`
        + `<span class="verdict-val">${v}</span></div>`
    ).join("");
    host.hidden = false;
    // The race itself is silent (the grid is rebuilt ten times a second); this
    // is the one announcement, and it carries the outcome.
    announce("Battle complete. " + rows.map(([k, v]) => k + ": " + stripTags(v)).join(". ") + ".");
  }

  // The verdict values are already-escaped HTML fragments; an announcement is
  // plain text.
  function stripTags(html) {
    return String(html)
      .replace(/<[^>]*>/g, "")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'");
  }

  async function streamContender(entry, body, onTick) {
    const started = performance.now();
    entry.state = "waiting";
    onTick();
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 401) { showAuthBanner(); revealKeyField(); }
      throw new Error(data && data.error ? data.error.message : "HTTP " + resp.status);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const evt of events) {
        const line = evt.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let obj;
        try { obj = JSON.parse(payload); } catch { continue; }
        if (obj.error) throw new Error(obj.error.message || "stream error");
        if (obj.usage) {
          entry.usage = obj.usage;
          // The server prices the run and reports it here, so the tally never
          // disagrees with the ledger.
          if (obj.effgen && obj.effgen.cost_usd != null) entry.cost = obj.effgen.cost_usd;
        }
        const delta = (obj.choices && obj.choices[0] && obj.choices[0].delta) || {};
        if (delta.content) {
          if (entry.ttft == null) entry.ttft = performance.now() - started;
          entry.state = "streaming";
          entry.text += delta.content;
          onTick();
        }
      }
    }
    entry.latency = performance.now() - started;
    entry.state = "done";
    onTick();
  }

  async function runBattle() {
    hideError();
    const prompt = $("prompt").value.trim();
    if (!prompt) { showError("Enter a prompt first."); return; }
    const models = selectedContenders();
    if (models.length < 2) {
      showError("Pick at least two contenders — a battle compares models against each other.");
      return;
    }
    if (!apiKey && !bootstrap.dev_mode) { showAuthBanner(); revealKeyField(); return; }

    const temperature = parseFloat($("temperature").value);
    const maxTokens = parseInt($("max-tokens").value, 10);
    const system = currentSystemPrompt();
    const entries = models.map((m) => ({
      model: m, text: "", state: "queued", ttft: null, latency: null,
      usage: null, cost: null, error: null,
    }));

    const btn = $("run-btn");
    btn.disabled = true;
    btn.textContent = "Racing…";
    $("battle-verdict").hidden = true;
    const grid = $("battle-grid");
    if (grid) grid.setAttribute("aria-busy", "true");
    renderBattleColumns(entries);

    // One frame per animation tick rather than per token, so a fast model does
    // not force a re-render of every column on every delta.
    let dirty = false;
    const tick = () => { dirty = true; };
    const timer = setInterval(() => {
      if (!dirty) return;
      dirty = false;
      renderBattleColumns(entries);
    }, 100);

    const started = performance.now();
    await Promise.all(entries.map((entry) => {
      const body = {
        model: entry.model,
        messages: buildMessages(system, prompt),
        temperature: isNaN(temperature) ? undefined : temperature,
        max_tokens: isNaN(maxTokens) ? undefined : maxTokens,
        stream: true,
        stream_options: { include_usage: true },
      };
      // One contender failing leaves the rest of the field racing.
      return streamContender(entry, body, tick).catch((err) => {
        entry.error = String(err && err.message ? err.message : err);
        entry.state = "failed";
      });
    }));
    clearInterval(timer);
    renderBattleColumns(entries);
    if (grid) grid.setAttribute("aria-busy", "false");
    renderBattleVerdict(entries, performance.now() - started);
    btn.disabled = false;
    syncMode();
  }

  // ------------------------------------------------------------------
  // Snippets
  // ------------------------------------------------------------------
  function shellQuote(s) { return "'" + String(s).replace(/'/g, "'\\''") + "'"; }
  function pyQuote(s) { return JSON.stringify(String(s)); }

  function renderSnippet() {
    if (!lastRun) return;
    const box = $("snippet-code");
    box.textContent = buildSnippet(snippetKind, lastRun);
  }

  function buildSnippet(kind, r) {
    const preset = $("preset-select").value;
    if (kind === "curl") {
      const body = {
        model: r.model,
        messages: buildMessages(r.system, r.prompt),
      };
      if (!isNaN(r.temperature)) body.temperature = r.temperature;
      if (!isNaN(r.maxTokens)) body.max_tokens = r.maxTokens;
      if (r.tools.length) body.tools = toolSpecs(r.tools);
      return "curl http://127.0.0.1:8000/v1/chat/completions \\\n"
        + "  -H 'Authorization: Bearer $EFFGEN_API_KEY' \\\n"
        + "  -H 'Content-Type: application/json' \\\n"
        + "  -d " + shellQuote(JSON.stringify(body));
    }
    if (kind === "cli") {
      let cmd = "effgen run " + shellQuote(r.prompt) + " -m " + shellQuote(r.model);
      if (preset) cmd += " --preset " + preset;
      if (r.tools.length) cmd += " -t " + r.tools.join(" ");
      if (!isNaN(r.temperature)) cmd += " --temperature " + r.temperature;
      if (!isNaN(r.maxTokens)) cmd += " --max-tokens " + r.maxTokens;
      return cmd;
    }
    // python
    const hasTools = r.tools.length > 0;
    const lines = ["from effgen import Agent", "from effgen.core.agent import AgentConfig"];
    if (hasTools) lines.push("from effgen.tools import get_registry");
    lines.push("");
    if (hasTools) { lines.push("reg = get_registry()"); lines.push(""); }
    const cfg = ["    model=" + pyQuote(r.model)];
    // AgentConfig(tools=...) takes Tool instances, so resolve each name from the
    // registry rather than passing bare strings.
    if (hasTools) {
      const resolved = r.tools.map((n) => "reg.get_tool_sync(" + pyQuote(n) + ")").join(", ");
      cfg.push("    tools=[" + resolved + "]");
    }
    if (!isNaN(r.temperature)) cfg.push("    temperature=" + r.temperature);
    if (!isNaN(r.maxTokens)) cfg.push("    max_tokens=" + r.maxTokens);
    if (r.system && r.system.trim()) cfg.push("    system_prompt=" + pyQuote(r.system));
    lines.push("agent = Agent(AgentConfig(");
    lines.push(cfg.join(",\n"));
    lines.push("))");
    lines.push("print(agent.run(" + pyQuote(r.prompt) + "))");
    return lines.join("\n");
  }

  // The three snippet tabs are one tab stop: the selected tab holds it and the
  // arrow keys move between them, per the ARIA tabs pattern.
  function selectSnippetTab(kind, moveFocus) {
    const tabs = Array.from(document.querySelectorAll(".snippet-tab"));
    const chosen = tabs.find((t) => t.dataset.kind === kind) || tabs[0];
    if (!chosen) return;
    tabs.forEach((t) => {
      const on = t === chosen;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.setAttribute("tabindex", on ? "0" : "-1");
    });
    const panel = $("snippet-panel");
    if (panel) panel.setAttribute("aria-labelledby", chosen.id);
    snippetKind = chosen.dataset.kind;
    renderSnippet();
    if (moveFocus) chosen.focus();
  }

  function initSnippetTabs() {
    const tabs = Array.from(document.querySelectorAll(".snippet-tab"));
    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => selectSnippetTab(tab.dataset.kind, false));
      tab.addEventListener("keydown", (e) => {
        const delta = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (delta) {
          e.preventDefault();
          const next = tabs[(i + delta + tabs.length) % tabs.length];
          selectSnippetTab(next.dataset.kind, true);
        } else if (e.key === "Home" || e.key === "End") {
          e.preventDefault();
          selectSnippetTab(tabs[e.key === "Home" ? 0 : tabs.length - 1].dataset.kind, true);
        }
      });
    });
    const copy = $("copy-btn");
    if (copy) copy.addEventListener("click", () => copySnippet());
  }

  async function copySnippet() {
    const box = $("snippet-code");
    const copy = $("copy-btn");
    if (!box) return;
    try {
      await navigator.clipboard.writeText(box.textContent);
      if (copy) {
        copy.textContent = "Copied";
        setTimeout(() => { copy.textContent = "Copy"; }, 1500);
      }
    } catch {
      // Clipboard blocked (non-secure context): select the text as a fallback.
      const range = document.createRange();
      range.selectNodeContents(box);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  }

  // ------------------------------------------------------------------
  // Keyboard navigation: command palette + document-level submit
  //
  // Commands are rebuilt each time the palette opens, so the model list
  // reflects whatever the catalog fetch has loaded by then.
  // ------------------------------------------------------------------
  function focusControl(id) {
    const el = $(id);
    if (!el) return;
    if (webui) webui.jumpTo("main-content");
    el.focus();
  }

  function snippetCommands() {
    return [
      ["curl", "Copy this run as a curl command"],
      ["cli", "Copy this run as an effgen CLI command"],
      ["python", "Copy this run as Python"],
    ].map(([kind, label]) => ({
      id: "snippet:" + kind,
      group: "Copy",
      label: label,
      keywords: "snippet copy clipboard " + kind,
      hint: kind,
      run: () => {
        if (!lastRun) { showError("Run a prompt first — the snippet copies that request."); return; }
        selectSnippetTab(kind, false);
        copySnippet();
      },
    }));
  }

  function modelCommands() {
    const sel = $("model-select");
    if (!sel) return [];
    return Array.from(sel.options).slice(0, 400).map((opt) => ({
      id: "model:" + opt.value,
      group: "Models",
      label: opt.value || opt.textContent,
      keywords: opt.textContent + " " + (opt.parentNode && opt.parentNode.label || ""),
      hint: (opt.parentNode && opt.parentNode.label) || "",
      run: () => {
        sel.value = opt.value;
        onModelChange();
        sel.focus();
      },
    }));
  }

  function presetCommands() {
    const sel = $("preset-select");
    if (!sel) return [];
    return Array.from(sel.options).map((opt) => ({
      id: "preset:" + (opt.value || "none"),
      group: "Presets",
      label: opt.value ? "Use the " + opt.value + " preset" : "Clear the preset",
      keywords: opt.textContent,
      run: () => {
        sel.value = opt.value;
        onPresetChange();
        sel.focus();
      },
    }));
  }

  function actionCommands() {
    const stream = $("stream-toggle");
    return [
      {
        id: "action:run",
        group: "Actions",
        label: currentMode() === "battle" ? "Start the battle" : "Run the request",
        keywords: "submit send prompt execute race battle",
        hint: webui ? webui.chord("Enter") : "",
        run: () => { if (currentMode() === "battle") runBattle(); else run(); },
      },
      {
        id: "action:mode",
        group: "Actions",
        label: currentMode() === "battle"
          ? "Switch to a single run"
          : "Switch to a battle — several models on one prompt",
        keywords: "battle compare head to head race mode single",
        run: () => {
          const next = currentMode() === "battle" ? "mode-single" : "mode-battle";
          const el = $(next);
          if (el) { el.checked = true; syncMode(); }
        },
      },
      {
        id: "action:focus-prompt",
        group: "Actions",
        label: "Edit the prompt",
        keywords: "prompt compose write task",
        run: () => focusControl("prompt"),
      },
      {
        id: "action:stream",
        group: "Actions",
        label: (stream && stream.checked ? "Turn off" : "Turn on") + " token streaming",
        keywords: "stream live tokens",
        run: () => { if (stream) stream.checked = !stream.checked; },
      },
      {
        id: "action:theme",
        group: "Actions",
        label: "Switch color theme",
        keywords: "dark light appearance",
        run: () => applyTheme(currentTheme() === "dark" ? "light" : "dark", true),
      },
      {
        id: "nav:dashboard",
        group: "Navigate",
        label: "Open the dashboard",
        keywords: "metrics runs history catalog topology",
        hint: "/dashboard",
        run: () => { window.location.href = "/dashboard"; },
      },
    ];
  }

  function initKeyboard() {
    // The submit shortcut lives on the document, so it works from the
    // temperature and max-token fields as well as the prompt box. An open
    // overlay owns Enter, so the shortcut stays out of the way there —
    // otherwise it would send a request alongside the chosen command.
    document.addEventListener("keydown", (e) => {
      if (!(e.ctrlKey || e.metaKey) || e.key !== "Enter") return;
      if (e.target && e.target.closest && e.target.closest(".eff-overlay")) return;
      e.preventDefault();
      if (currentMode() === "battle") runBattle(); else run();
    });
    if (!webui) return;
    const palette = webui.init({
      surface: "playground",
      actions: () => actionCommands()
        .concat(presetCommands(), snippetCommands(), modelCommands()),
      shortcuts: [
        { keys: webui.chord("Enter"), label: "Run the request from anywhere on the page" },
        { keys: "← →", label: "Move between the snippet tabs" },
      ],
    });
    const btn = $("palette-btn");
    if (btn) btn.addEventListener("click", palette.open);
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  async function init() {
    initTheme();
    initSnippetTabs();
    initKeyboard();
    $("run-btn").addEventListener("click", () => {
      if (currentMode() === "battle") runBattle(); else run();
    });
    ["mode-single", "mode-battle"].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("change", syncMode);
    });
    syncMode();
    await loadBootstrap();
    await loadCatalog();
    populateBattleModels();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
