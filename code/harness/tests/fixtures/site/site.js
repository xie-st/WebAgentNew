/* Deterministic local fixture data. No network, no external assets. */
(function () {
  "use strict";

  const WORDS = [
    "harbor", "meadow", "signal", "cobalt", "willow", "ember",
    "orchard", "lantern", "cinder", "tidal", "archive", "quartz"
  ];

  function rngFromSeed(seed) {
    let value = 2166136261;
    const text = String(seed || "seed");
    for (let i = 0; i < text.length; i += 1) {
      value ^= text.charCodeAt(i);
      value = Math.imul(value, 16777619);
    }
    return function next() {
      value |= 0;
      value = (value + 0x6D2B79F5) | 0;
      let t = Math.imul(value ^ (value >>> 15), 1 | value);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function catalog(seed, query) {
    const rng = rngFromSeed(seed + "|" + query);
    const count = 12;
    const items = [];
    for (let i = 0; i < count; i += 1) {
      const word = WORDS[Math.floor(rng() * WORDS.length)];
      const tone = Math.floor(rng() * 900) + 100;
      items.push({
        id: i + 1,
        title: query + " field note " + (i + 1) + ": " + word + " " + tone,
        summary:
          "A deterministic " + query + " record generated from seed " + seed +
          ". Topic word: " + word + ". Reference: " + tone + ".",
        details:
          "This detail page belongs to " + query + " result " + (i + 1) +
          ". The seed is " + seed + " and the generated marker is " + word + "-" + tone +
          ". It exists only in the local fixture site."
      });
    }
    return items;
  }

  function params() {
    return new URLSearchParams(window.location.search);
  }

  function seedValue() {
    return params().get("seed") || "local-seed";
  }

  function queryValue() {
    return (params().get("q") || "").trim() || "seed";
  }

  function renderHome() {
    const input = document.getElementById("query");
    const form = document.getElementById("search-form");
    const seedLine = document.getElementById("seed-line");
    const seed = seedValue();
    if (seedLine) seedLine.textContent = "Fixture seed: " + seed;
    if (form) {
      form.addEventListener("submit", function (event) {
        event.preventDefault();
        const q = (input && input.value ? input.value : "").trim();
        window.location.href =
          "results.html?seed=" + encodeURIComponent(seed) +
          "&q=" + encodeURIComponent(q || "seed") + "&page=1";
      });
    }
  }

  function renderResults() {
    const seed = seedValue();
    const query = queryValue();
    const page = Math.max(1, parseInt(params().get("page") || "1", 10) || 1);
    const pageSize = 5;
    const items = catalog(seed, query);
    const start = (page - 1) * pageSize;
    const pageItems = items.slice(start, start + pageSize);
    const heading = document.getElementById("results-heading");
    const summary = document.getElementById("results-summary");
    const list = document.getElementById("results-list");
    const pager = document.getElementById("pager");
    if (heading) heading.textContent = "Search results for \"" + query + "\"";
    if (summary) {
      summary.textContent =
        items.length + " result(s); page " + page + " of 3; seed " + seed + ".";
    }
    if (list) {
      list.textContent = "";
      for (const item of pageItems) {
        const li = document.createElement("li");
        const link = document.createElement("a");
        link.href =
          "detail.html?seed=" + encodeURIComponent(seed) +
          "&q=" + encodeURIComponent(query) + "&id=" + item.id;
        link.textContent = item.title;
        const note = document.createElement("p");
        note.className = "muted";
        note.textContent = item.summary;
        li.appendChild(link);
        li.appendChild(note);
        list.appendChild(li);
      }
    }
    if (pager) {
      pager.textContent = "";
      const previous = document.createElement("a");
      previous.href =
        "results.html?seed=" + encodeURIComponent(seed) +
        "&q=" + encodeURIComponent(query) + "&page=" + (page - 1);
      previous.textContent = "Previous page";
      if (page > 1) pager.appendChild(previous);
      const next = document.createElement("a");
      next.href =
        "results.html?seed=" + encodeURIComponent(seed) +
        "&q=" + encodeURIComponent(query) + "&page=" + (page + 1);
      next.textContent = "Next page";
      if (page < 3) pager.appendChild(next);
    }
  }

  function renderDetail() {
    const seed = seedValue();
    const query = queryValue();
    const id = Math.max(1, parseInt(params().get("id") || "1", 10) || 1);
    const item = catalog(seed, query)[id - 1] || catalog(seed, query)[0];
    const heading = document.getElementById("detail-heading");
    const summary = document.getElementById("detail-summary");
    const details = document.getElementById("detail-body");
    const back = document.getElementById("back-results");
    if (heading) heading.textContent = item.title;
    if (summary) summary.textContent = item.summary;
    if (details) details.textContent = item.details;
    if (back) {
      back.href =
        "results.html?seed=" + encodeURIComponent(seed) +
        "&q=" + encodeURIComponent(query) + "&page=1";
    }
    const previewButton = document.getElementById("preview-button");
    const draft = document.getElementById("draft");
    const preview = document.getElementById("preview");
    if (previewButton) {
      previewButton.addEventListener("click", function () {
        const text = draft && draft.value ? draft.value : "";
        if (preview) preview.textContent = "Preview: " + text;
      });
    }
  }

  const page = document.body.getAttribute("data-page");
  if (page === "home") renderHome();
  if (page === "results") renderResults();
  if (page === "detail") renderDetail();
})();
