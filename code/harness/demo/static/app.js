/* WebAgent 演示面板前端：短轮询 + 原生 DOM 渲染。
   所有动态内容一律用 textContent 渲染，防止 HTML 注入。 */
"use strict";

(function () {
  var $ = function (id) { return document.getElementById(id); };

  var els = {
    modeBadge: $("mode-badge"),
    stateBadge: $("state-badge"),
    task: $("task"),
    url: $("url"),
    seconds: $("seconds"),
    modeNote: $("mode-note"),
    btnPreset: $("btn-preset"),
    btnStart: $("btn-start"),
    btnPause: $("btn-pause"),
    btnResume: $("btn-resume"),
    btnCancel: $("btn-cancel"),
    btnForce: $("btn-force"),
    btnCopy: $("btn-copy"),
    formMessage: $("form-message"),
    cfgPath: $("cfg-path"),
    cfgModel: $("cfg-model"),
    stRunId: $("st-runid"),
    stMode: $("st-mode"),
    stStart: $("st-start"),
    stElapsed: $("st-elapsed"),
    stState: $("st-state"),
    stUrl: $("st-url"),
    stAction: $("st-action"),
    shotImg: $("shot-img"),
    shotEmpty: $("shot-empty"),
    shotCaption: $("shot-caption"),
    resultState: $("result-state"),
    answer: $("answer"),
    resTerm: $("res-term"),
    resVerify: $("res-verify"),
    log: $("log"),
    logEmpty: $("log-empty"),
    stdout: $("stdout"),
    stderr: $("stderr"),
    runDir: $("run-dir"),
    downloads: $("downloads"),
    shotList: $("shot-list"),
    historyLine: $("history-line"),
    lightbox: $("lightbox"),
    lightboxImg: $("lightbox-img"),
    lightboxCaption: $("lightbox-caption")
  };

  var lastRunId = null;
  var lastScreenshotFile = null;
  var serverInfo = null;
  var elapsedTimer = null;

  function setText(el, text) { el.textContent = text; }

  function fmtTime(epochSeconds) {
    if (!epochSeconds) return "暂无";
    var d = new Date(epochSeconds * 1000);
    function p(n) { return (n < 10 ? "0" : "") + n; }
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) +
      " " + p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  function fmtElapsed(seconds) {
    if (seconds == null) return "暂无";
    var s = Math.max(0, Math.floor(seconds));
    var m = Math.floor(s / 60);
    s = s % 60;
    var h = Math.floor(m / 60);
    m = m % 60;
    return (h ? h + " 时 " : "") + (h || m ? m + " 分 " : "") + s + " 秒";
  }

  // ---------- 消息 ----------
  function message(text, ok) {
    setText(els.formMessage, text || "");
    els.formMessage.classList.toggle("info", !!ok);
  }

  // ---------- API ----------
  function api(url, options) {
    return fetch(url, options).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (!res.ok) throw new Error(data.error || ("请求失败（HTTP " + res.status + "）"));
        return data;
      });
    });
  }

  // ---------- 模式 ----------
  var modeRadios = function () {
    return Array.prototype.slice.call(document.querySelectorAll('input[name="mode"]'));
  };
  function currentMode() {
    var checked = modeRadios().filter(function (r) { return r.checked; })[0];
    return checked ? checked.value : "real";
  }
  function setMode(mode) {
    modeRadios().forEach(function (r) { r.checked = r.value === mode; });
    applyMode();
  }
  function applyMode() {
    var mode = currentMode();
    var offline = mode === "offline";
    els.task.disabled = offline;
    els.url.disabled = offline;
    if (offline) {
      if (serverInfo && serverInfo.preset) {
        els.task.value = serverInfo.preset.task;
        els.url.value = serverInfo.preset.start_url;
      }
    }
    renderModeBadge();
    saveInputs();
  }
  function renderModeBadge() {
    var mode = currentMode();
    if (!serverInfo) return;
    var m = serverInfo.mode;
    if (mode === "offline") {
      setText(els.modeBadge, "离线脚本演示：固定响应，未调用真实模型");
      return;
    }
    if (m.available) {
      setText(els.modeBadge, "真实模型" + (m.model_name ? "（" + m.model_name + "）" : ""));
    } else {
      setText(els.modeBadge, "真实模型不可用");
    }
  }

  // ---------- 输入记忆 ----------
  function saveInputs() {
    try {
      if (currentMode() === "real") {
        localStorage.setItem("demo.task", els.task.value);
        localStorage.setItem("demo.url", els.url.value);
      }
      localStorage.setItem("demo.seconds", els.seconds.value);
    } catch (e) { /* 忽略存储失败 */ }
  }
  function restoreInputs() {
    try {
      var t = localStorage.getItem("demo.task");
      var u = localStorage.getItem("demo.url");
      var s = localStorage.getItem("demo.seconds");
      if (t) els.task.value = t;
      if (u) els.url.value = u;
      if (s) els.seconds.value = s;
    } catch (e) { /* 忽略 */ }
  }

  // ---------- 渲染 ----------
  var ACTIVE_STATES = ["starting", "running", "pause_requested", "paused",
    "resume_requested", "cancel_requested"];

  function render(data) {
    serverInfo = data;
    var run = data.run;
    renderModeBadge();
    if (data.mode && !data.mode.available) {
      setText(els.modeNote, data.mode.reason || "");
      var real = modeRadios().filter(function (r) { return r.value === "real"; })[0];
      real.disabled = true;
      if (currentMode() === "real") setMode("offline");
    } else {
      setText(els.modeNote, "");
      modeRadios().forEach(function (r) { r.disabled = false; });
    }
    if (data.mode) {
      setText(els.cfgPath, data.mode.config_path || "—");
      setText(els.cfgModel, data.mode.model_name || "—");
    }

    // 状态徽章
    var state = run ? run.state : "idle";
    var label = run ? run.state_label : "待运行";
    setText(els.stateBadge, label);
    els.stateBadge.setAttribute("data-state", state);
    if (state === "ended" && run.outcome_label) {
      els.stateBadge.setAttribute("data-outcome", run.outcome);
      setText(els.stateBadge, "已结束：" + run.outcome_label);
    } else {
      els.stateBadge.removeAttribute("data-outcome");
    }

    // 按钮
    var active = run && ACTIVE_STATES.indexOf(run.state) >= 0;
    els.btnStart.disabled = !!active;
    els.btnPause.disabled = !(run && (run.state === "running" || run.state === "starting"));
    els.btnResume.disabled = !(run && run.state === "paused");
    // 已暂停时“继续”成为主要操作。
    els.btnResume.classList.toggle("primary", !!(run && run.state === "paused"));
    els.btnCancel.disabled = !active;
    els.btnForce.disabled = !active;
    els.btnPreset.disabled = false;

    if (!run) {
      els.stRunId.textContent = "暂无";
      els.stMode.textContent = "暂无";
      els.stStart.textContent = "暂无";
      els.stElapsed.textContent = "暂无";
      els.stState.textContent = "待运行";
      els.stState.removeAttribute("data-tone");
      els.stUrl.textContent = "暂无";
      els.stAction.textContent = "暂无";
      els.shotImg.hidden = true;
      els.shotEmpty.hidden = false;
      els.shotCaption.textContent = "";
      els.resultState.textContent = "输入任务并点击“开始运行”后，这里展示最终回答与结束情况。";
      els.resultState.removeAttribute("data-tone");
      els.answer.textContent = "暂无";
      els.btnCopy.disabled = true;
      els.resTerm.textContent = "暂无";
      els.resVerify.textContent = "未提供本地核验结果";
      els.runDir.textContent = "暂无";
      els.downloads.innerHTML = "";
      els.shotList.innerHTML = "";
      els.log.innerHTML = "";
      els.logEmpty.hidden = false;
      els.stdout.textContent = "";
      els.stderr.textContent = "";
      els.historyLine.textContent = "";
      stopElapsed();
      return;
    }

    // 状态条
    lastRunId = run.run_id;
    setText(els.stRunId, run.run_id);
    setText(els.stMode, run.mode === "offline" ? "离线脚本演示" : "真实模型");
    setText(els.stStart, fmtTime(run.started_at));
    setText(els.stElapsed, fmtElapsed(run.elapsed_s));
    setText(els.stState, run.state === "ended" ? ("已结束：" + (run.outcome_label || "未知")) : run.state_label);
    var tone = { running: "run", starting: "run", paused: "wait", pause_requested: "wait",
      resume_requested: "wait", cancel_requested: "wait" }[run.state];
    if (run.state === "ended") {
      tone = run.outcome === "completed" ? "ok" :
        (run.outcome === "cancelled" || run.outcome === "interrupted") ? "wait" : "bad";
    }
    if (tone) els.stState.setAttribute("data-tone", tone); else els.stState.removeAttribute("data-tone");
    setText(els.stUrl, run.current_url || "暂无");
    setText(els.stAction, run.last_action || "暂无");

    // 截图
    var shots = run.screenshots || [];
    if (shots.length) {
      var latest = shots[shots.length - 1];
      if (latest.file !== lastScreenshotFile) {
        lastScreenshotFile = latest.file;
        els.shotImg.src = "/api/screenshot?run=" + encodeURIComponent(run.run_id) +
          "&file=" + encodeURIComponent(latest.file);
      }
      els.shotImg.hidden = false;
      els.shotEmpty.hidden = true;
      setText(els.shotCaption,
        "run_id：" + run.run_id + " · 文件：" + latest.file +
        (latest.observation_id ? " · 观察：" + latest.observation_id : "") +
        (latest.url ? " · " + latest.url : ""));
    } else {
      els.shotImg.hidden = true;
      els.shotEmpty.hidden = false;
      els.shotCaption.textContent = run.state === "ended" ? "本次运行没有产生截图" : "";
      lastScreenshotFile = null;
    }

    // 结果
    if (run.state === "ended") {
      var ok = run.outcome === "completed";
      els.resultState.setAttribute("data-tone", ok ? "ok" :
        (run.outcome === "cancelled" || run.outcome === "interrupted") ? "wait" : "bad");
      els.resultState.textContent = "本次任务已结束：" + (run.outcome_label || "未知") +
        (run.forced ? "（由面板强制结束）" : "");
    } else {
      els.resultState.removeAttribute("data-tone");
      els.resultState.textContent = "任务执行中：" + run.state_label;
    }
    if (run.answer) {
      els.answer.textContent = run.answer;
      els.btnCopy.disabled = false;
    } else if (run.state === "ended") {
      els.answer.textContent = "未返回最终回答";
      els.btnCopy.disabled = true;
    } else {
      els.answer.textContent = "暂无";
      els.btnCopy.disabled = true;
    }
    var term = run.termination_label ||
      (run.unfinished ? String(run.unfinished) : null) ||
      (run.state === "ended" ? "结果文件缺失或未产生终止原因" : "暂无");
    setText(els.resTerm, term);
    setText(els.resVerify, run.verification_text || "未提供本地核验结果");

    // 日志
    els.log.innerHTML = "";
    (run.events || []).forEach(function (e) {
      var li = document.createElement("li");
      li.setAttribute("data-kind", e.kind || "");
      var t = document.createElement("span");
      t.className = "t";
      t.textContent = (e.time || "") + (e.elapsed_s != null ? " +" + e.elapsed_s + "s" : "");
      var x = document.createElement("span");
      x.className = "x";
      x.textContent = e.text || "";
      li.appendChild(t);
      li.appendChild(x);
      els.log.appendChild(li);
    });
    els.logEmpty.hidden = (run.events || []).length > 0;
    els.stdout.textContent = (run.stdout_tail || []).join("\n");
    els.stderr.textContent = (run.stderr_tail || []).join("\n");

    // 证据
    setText(els.runDir, run.run_dir);
    els.downloads.innerHTML = "";
    var names = { "task.json": "任务输入", "result.json": "结果 JSON", "events.jsonl": "事件日志",
      "stdout.log": "标准输出", "stderr.log": "标准错误", "meta.json": "面板元数据" };
    var files = run.files || [];
    if (files.length) {
      files.forEach(function (name) {
        var a = document.createElement("a");
        a.href = "/api/download?run=" + encodeURIComponent(run.run_id) +
          "&name=" + encodeURIComponent(name);
        a.textContent = names[name] || name;
        els.downloads.appendChild(a);
      });
    } else {
      var span = document.createElement("span");
      span.className = "none";
      span.textContent = "尚无可下载的文件";
      els.downloads.appendChild(span);
    }
    els.shotList.innerHTML = "";
    shots.slice().reverse().forEach(function (s) {
      var li = document.createElement("li");
      var btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = s.file;
      btn.addEventListener("click", function () { openLightbox(run.run_id, s); });
      li.appendChild(btn);
      var info = document.createTextNode(
        (s.time ? " " + s.time : "") + (s.url ? " · " + s.url : ""));
      li.appendChild(info);
      els.shotList.appendChild(li);
    });
    els.historyLine.textContent =
      (data.history || []).length ? ("本服务已启动的运行：" + data.history.join("、")) : "";

    if (run.state === "ended") stopElapsed(); else startElapsed();
  }

  // ---------- 已用时间 ----------
  function startElapsed() {
    if (elapsedTimer) return;
    elapsedTimer = setInterval(tickElapsed, 1000);
  }
  function stopElapsed() {
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
  }
  function tickElapsed() {
    // 仅在运行中时本地刷新显示，权威值仍来自服务端。
    if (!serverInfo || !serverInfo.run) return;
    var run = serverInfo.run;
    if (run.state === "ended") { stopElapsed(); return; }
    var started = run.started_at_ms || (run.started_at ? run.started_at * 1000 : null);
    if (!started) return;
    els.stElapsed.textContent = fmtElapsed((Date.now() - started) / 1000);
  }

  // ---------- 放大 ----------
  function openLightbox(runId, shot) {
    els.lightboxImg.src = "/api/screenshot?run=" + encodeURIComponent(runId) +
      "&file=" + encodeURIComponent(shot.file);
    els.lightboxCaption.textContent =
      "run_id：" + runId + " · 文件：" + shot.file +
      (shot.observation_id ? " · 观察：" + shot.observation_id : "") +
      (shot.url ? " · " + shot.url : "");
    if (typeof els.lightbox.showModal === "function") els.lightbox.showModal();
  }

  // ---------- 事件 ----------
  els.btnPreset.addEventListener("click", function () {
    if (serverInfo && serverInfo.preset) {
      els.task.value = serverInfo.preset.task;
      els.url.value = serverInfo.preset.start_url;
      saveInputs();
      message("已填入本地预设任务。", true);
    }
  });

  modeRadios().forEach(function (r) {
    r.addEventListener("change", applyMode);
  });

  els.task.addEventListener("change", saveInputs);
  els.url.addEventListener("change", saveInputs);
  els.seconds.addEventListener("change", saveInputs);

  $("start-form").addEventListener("submit", function (event) {
    event.preventDefault();
    if (els.btnStart.disabled) return;
    var seconds = Number(els.seconds.value);
    message("");
    els.btnStart.disabled = true;
    api("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: currentMode(),
        task: els.task.value,
        start_url: els.url.value,
        max_seconds: seconds
      })
    }).then(function (data) {
      message("已开始运行（run_id：" + data.run_id + "）。", true);
      poll();
    }).catch(function (err) {
      message(err.message);
      els.btnStart.disabled = false;
    });
  });

  function control(command) {
    message("");
    api("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: command })
    }).then(function () {
      if (command === "pause") message("已提交暂停请求，等待当前动作结束…", true);
      if (command === "resume") message("已提交继续请求，等待重新观察页面…", true);
      if (command === "cancel") message("已提交取消请求，等待任务结束…", true);
      poll();
    }).catch(function (err) { message(err.message); poll(); });
  }
  els.btnPause.addEventListener("click", function () { control("pause"); });
  els.btnResume.addEventListener("click", function () { control("resume"); });
  els.btnCancel.addEventListener("click", function () { control("cancel"); });

  els.btnForce.addEventListener("click", function () {
    if (!window.confirm("确定强制结束本次任务吗？执行进程会被终止，已有证据保留，结果记为中断。")) return;
    message("");
    api("/api/force-end", { method: "POST" })
      .then(function () { message("已发送强制结束，正在等待进程退出…", true); poll(); })
      .catch(function (err) { message(err.message); poll(); });
  });

  els.btnCopy.addEventListener("click", function () {
    var text = els.answer.textContent || "";
    if (!text || text === "暂无") return;
    function done() {
      var old = els.btnCopy.textContent;
      els.btnCopy.textContent = "已复制";
      setTimeout(function () { els.btnCopy.textContent = old; }, 1200);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
    } else {
      fallbackCopy(text); done();
    }
  });
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* 忽略 */ }
    document.body.removeChild(ta);
  }

  els.shotImg.addEventListener("click", function () {
    if (!serverInfo || !serverInfo.run || els.shotImg.hidden) return;
    var shots = serverInfo.run.screenshots || [];
    var latest = shots[shots.length - 1];
    if (latest) openLightbox(serverInfo.run.run_id, latest);
  });

  // ---------- 轮询 ----------
  var polling = false;
  function poll() {
    if (polling) return;
    polling = true;
    api("/api/state").then(function (data) {
      render(data);
    }).catch(function (err) {
      message("无法连接面板服务：" + err.message);
    }).finally(function () {
      polling = false;
      scheduleNext();
    });
  }
  function scheduleNext() {
    var active = serverInfo && serverInfo.run && serverInfo.run.state !== "ended";
    setTimeout(poll, active ? 1000 : 2500);
  }

  restoreInputs();
  poll();
})();
