(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    projectId: null,
    snapshot: null,
    projects: [],
    stream: null,
    busy: false,
    artifactCache: new Map(),
    openArtifacts: new Set(),
    renderedMessageIds: "",
    newMessageIds: new Set(),
    notificationsRequested: false,
    messageNodes: new Map(),
    programName: null,
  };
  const baseTitle = document.title;
  let titleFlashTimer = null;

  const els = {
    sidebar: $("#sidebar"), list: $("#project-list"), conversation: $("#conversation"),
    welcome: $("#welcome"), title: $("#project-title"), progress: $("#project-progress"),
    status: $("#status-pill"), prompt: $("#prompt"), context: $("#context"),
    composer: $("#composer"), send: $("#send"), toast: $("#toast"),
    connectionDot: $("#connection-dot"), connectionLabel: $("#connection-label"),
  };

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
    })[char]);
  }

  function activityTitle(activity) {
    const title = escapeHtml(activity.title);
    const url = String(activity.source_url || "");
    return /^https?:\/\/[^\s]+$/i.test(url)
      ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${title}</a>`
      : title;
  }

  function markdown(text) {
    const lines = escapeHtml(text).split(/\r?\n/);
    let html = "", list = false;
    const inline = (line) => line
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    const tableCells = (line) => {
      let value = line.trim();
      if (value.startsWith("|")) value = value.slice(1);
      if (value.endsWith("|")) value = value.slice(0, -1);
      return value.split("|").map((cell) => cell.trim());
    };
    const tableSeparator = (line) => {
      if (!line.includes("|")) return false;
      const cells = tableCells(line);
      return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
    };
    const closeList = () => {
      if (list) { html += "</ul>"; list = false; }
    };
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      if (line.trim().startsWith("```")) {
        closeList();
        const code = [];
        while (++index < lines.length && !lines[index].trim().startsWith("```")) code.push(lines[index]);
        html += `<pre><code>${code.join("\n")}</code></pre>`;
        continue;
      }
      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) { closeList(); html += `<h${heading[1].length + 1}>${inline(heading[2])}</h${heading[1].length + 1}>`; continue; }
      if (
        line.includes("|")
        && index + 1 < lines.length
        && tableSeparator(lines[index + 1])
      ) {
        closeList();
        const headers = tableCells(line);
        const alignments = tableCells(lines[index + 1]).map((cell) => {
          if (cell.startsWith(":") && cell.endsWith(":")) return "center";
          if (cell.endsWith(":")) return "right";
          return "left";
        });
        const rows = [];
        index += 2;
        while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
          rows.push(tableCells(lines[index]));
          index += 1;
        }
        index -= 1;
        html += '<div class="message-table-wrap"><table class="data-table message-table"><thead><tr>';
        html += headers.map((cell, column) =>
          `<th style="text-align:${alignments[column] || "left"}">${inline(cell)}</th>`
        ).join("");
        html += "</tr></thead><tbody>";
        html += rows.map((row) => `<tr>${headers.map((_, column) =>
          `<td style="text-align:${alignments[column] || "left"}">${inline(row[column] || "")}</td>`
        ).join("")}</tr>`).join("");
        html += "</tbody></table></div>";
        continue;
      }
      const item = line.match(/^\s*[-*]\s+(.+)$/);
      if (item) {
        if (!list) { html += "<ul>"; list = true; }
        html += `<li>${inline(item[1])}</li>`;
      } else {
        closeList();
        if (line.trim()) html += `<p>${inline(line)}</p>`;
      }
    }
    closeList();
    return html;
  }

  function deduplicateAttachments(items = []) {
    const selected = new Map();
    items.forEach((attachment, index) => {
      if (!["image", "download"].includes(attachment.type)) {
        selected.set(`other:${index}`, attachment);
        return;
      }
      // Similar titles and equal file sizes are not evidence of duplication:
      // before/after optics can share both while showing different candidates.
      const identity = attachment.content_sha256 || attachment.url || attachment.path;
      const key = identity ? `${attachment.type}:${identity}` : `other:${index}`;
      if (!selected.has(key)) selected.set(key, attachment);
    });
    return [...selected.values()];
  }

  function formatValue(value, digits = 5) {
    if (value === null || value === undefined || value === "") return "n/a";
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    if (number !== 0 && (Math.abs(number) >= 1e5 || Math.abs(number) < 1e-3)) {
      return number.toExponential(3);
    }
    return number.toLocaleString(undefined, { maximumFractionDigits: digits });
  }

  function timeLabel(timestamp) {
    const date = new Date(timestamp);
    return Number.isNaN(date.valueOf()) ? "" : date.toLocaleString([], {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  }

  function modelLabel(backend = {}) {
    return backend.model || backend.name || "Configured model";
  }

  function networkLabel(network = {}) {
    const web = network.enabled ? "Web online" : "Web offline";
    return network.local_mcp_enabled ? `${web} · Local MCP` : web;
  }

  function elapsedLabel(seconds) {
    const total = Math.max(0, Number(seconds) || 0);
    if (total < 60) return `${Math.floor(total)} s`;
    const minutes = Math.floor(total / 60);
    const remainder = Math.floor(total % 60);
    return `${minutes} min ${remainder} s`;
  }

  async function api(url, options = {}) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try { detail = (await response.json()).detail || detail; } catch (_) { /* no-op */ }
      throw new Error(detail);
    }
    return response.json();
  }

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.classList.remove("hidden");
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => els.toast.classList.add("hidden"), 5000);
  }

  function collectImageAttachmentCount(message) {
    return (message.attachments || []).filter((item) => item.type === "image").length;
  }

  function ensureNotificationPermission() {
    if (state.notificationsRequested) return;
    state.notificationsRequested = true;
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission().catch(() => { /* no-op */ });
    }
  }

  function flashTitle(text) {
    clearInterval(titleFlashTimer);
    let showAlt = true;
    titleFlashTimer = setInterval(() => {
      document.title = showAlt ? `🔴 ${text}` : baseTitle;
      showAlt = !showAlt;
    }, 1200);
  }

  function stopTitleFlash() {
    if (!titleFlashTimer) return;
    clearInterval(titleFlashTimer);
    titleFlashTimer = null;
    document.title = baseTitle;
  }

  function sendDesktopNotification(text) {
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    try { new Notification("Brilliance research update", { body: text }); } catch (_) { /* no-op */ }
  }

  // The model does not always narrate every generated figure in its own prose,
  // and a scientist who is not watching the tab can otherwise miss new plots
  // entirely. Surface them immediately instead of relying on scrolling.
  function notifyNewPlots(count, projectTitle) {
    const text = `${count} new plot${count === 1 ? "" : "s"} generated in "${projectTitle}"`;
    showToast(text);
    if (document.hidden) {
      flashTitle("New research result — Brilliance");
      sendDesktopNotification(text);
    }
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    stopTitleFlash();
    state.newMessageIds = new Set();
  });

  function setConnected(connected, label = "Live updates connected") {
    els.connectionDot.classList.toggle("online", connected);
    els.connectionLabel.textContent = connected ? label : "Reconnecting live updates…";
  }

  async function loadProjects() {
    try {
      const data = await api("/api/projects");
      state.projects = data.projects || [];
      renderProjects();
    } catch (error) { showToast(error.message); }
  }

  function renderProjects() {
    els.list.replaceChildren(...state.projects.map((project) => {
      const button = document.createElement("button");
      button.className = `project-link${project.id === state.projectId ? " active" : ""}`;
      const accepted = project.progress?.accepted ?? 0;
      const total = project.progress?.total ?? 0;
      button.innerHTML = `<div class="title">${escapeHtml(project.title)}</div>
        <div class="meta"><span>${escapeHtml(project.status.replaceAll("_", " "))}</span><span>${escapeHtml(modelLabel(project.backend || {}))}</span></div>`;
      button.addEventListener("click", () => openProject(project.id));
      return button;
    }));
  }

  function resetToNew() {
    resetWorkspace();
    closeStream();
    state.projectId = null;
    state.snapshot = null;
    state.renderedMessageIds = "";
    history.pushState({}, "", "/");
    els.conversation.replaceChildren();
    els.conversation.classList.add("hidden");
    els.welcome.classList.remove("hidden");
    els.title.textContent = "New accelerator research";
    els.progress.textContent = "";
    els.status.classList.add("hidden");
    els.context.classList.add("hidden");
    els.context.value = "";
    els.prompt.value = "";
    renderProjects();
    autoResize();
    els.prompt.focus();
  }

  async function openProject(projectId, push = true) {
    if (!projectId) return resetToNew();
    resetWorkspace();
    closeStream();
    state.projectId = projectId;
    state.snapshot = null;
    state.renderedMessageIds = "";
    if (push) history.pushState({ projectId }, "", `/p/${encodeURIComponent(projectId)}`);
    els.welcome.classList.add("hidden");
    els.conversation.classList.remove("hidden");
    els.conversation.innerHTML = `<div class="artifact-loading"><span class="spinner"></span>Opening the research conversation…</div>`;
    renderProjects();
    try {
      applySnapshot(await api(`/api/projects/${encodeURIComponent(projectId)}`), true);
      startStream(projectId, state.snapshot?.version || "");
    } catch (error) {
      showToast(error.message);
      resetToNew();
    }
  }

  function closeStream() {
    if (state.stream) state.stream.close();
    state.stream = null;
    setConnected(false, "");
  }

  function startStream(projectId, version = "") {
    const stream = new EventSource(`/api/projects/${encodeURIComponent(projectId)}/events?version=${encodeURIComponent(version)}`);
    state.stream = stream;
    stream.addEventListener("snapshot", (event) => {
      if (state.projectId !== projectId) return;
      try { applySnapshot(JSON.parse(event.data)); } catch (error) { showToast(error.message); }
    });
    stream.addEventListener("delta", (event) => {
      if (state.projectId !== projectId || !state.snapshot) return;
      try {
        const delta = JSON.parse(event.data);
        const messages = new Map(state.snapshot.messages.map((message) => [message.id, message]));
        delta.upsert.forEach((message) => messages.set(message.id, message));
        applySnapshot({ ...delta, messages: delta.message_order.map((id) => messages.get(id)).filter(Boolean) });
      } catch (error) { showToast(error.message); }
    });
    stream.onopen = () => setConnected(true);
    stream.onerror = () => setConnected(false);
    stream.addEventListener("error", (event) => {
      if (event.data) {
        try { showToast(JSON.parse(event.data).detail); } catch (_) { /* browser reconnect */ }
      }
    });
  }

  function applySnapshot(snapshot, force = false) {
    const wasNearBottom = nearBottom();
    const previousMessages = force ? [] : (state.snapshot?.messages || []);
    const previousIds = new Set(previousMessages.map((item) => item.id));
    state.snapshot = snapshot;
    state.snapshotReceivedAt = Date.now();
    updateWorkspace(snapshot);
    const project = snapshot.project;
    els.title.textContent = project.title;
    const progress = project.progress || {};
    els.progress.textContent = `${progress.accepted || 0} of ${progress.total || 0} evidence gates accepted · revision ${project.revision}`;
    const computationRunning = (snapshot.computations || []).length > 0;
    els.status.className = `status-pill${snapshot.live || computationRunning ? " running" : ""}`;
    els.status.textContent = snapshot.live || computationRunning ? "JuTrack running" : project.status.replaceAll("_", " ");
    els.status.classList.remove("hidden");
    if (project.runtime === "codex") {
      const backend = snapshot.live?.backend || project.backend || {};
      const currentModel = modelLabel(backend);
      const researchNetwork = snapshot.live?.research_network || project.research_network || {};
      els.progress.textContent = `${currentModel} · JuTrack · ${networkLabel(researchNetwork)} · project revision ${project.revision}`;
      els.status.textContent = snapshot.live
        ? `${currentModel} working`
        : computationRunning
        ? "Numerical campaign running"
        : (backend.configured === false ? "model setup needed" : project.status.replaceAll("_", " "));
    }
    if (project.active_lattice_id) els.progress.textContent += ` · Working lattice: ${project.active_lattice_id}`;
    if (!snapshot.live && (snapshot.session?.protocol_recovery || snapshot.session?.pause_requested)) {
      els.status.textContent = computationRunning ? "Agent paused · calculation running" : "Research paused";
    }
    const signature = (snapshot.messages || []).map((message) => message.id).join("|") +
      `:${snapshot.live?.state || ""}:${snapshot.live?.version || ""}:${snapshot.version || ""}`;
    if (force || signature !== state.renderedMessageIds) {
      state.renderedMessageIds = signature;
      const newAssistantMessages = (snapshot.messages || []).filter(
        (message) => message.role === "assistant" && !previousIds.has(message.id)
      );
      const newImageCount = newAssistantMessages.reduce(
        (sum, message) => sum + collectImageAttachmentCount(message), 0
      );
      state.newMessageIds = new Set(newAssistantMessages.map((message) => message.id));
      if (!force && newImageCount > 0) notifyNewPlots(newImageCount, project.title);
      renderConversation(snapshot);
      if (wasNearBottom || force) requestAnimationFrame(() => scrollBottom(force));
    }
    loadProjects();
  }

  function renderConversation(snapshot) {
    const retained = new Set();
    const nodes = (snapshot.messages || []).map((message) => {
      retained.add(message.id);
      const signature = JSON.stringify(message);
      let cached = state.messageNodes.get(message.id);
      if (!cached || cached.signature !== signature) {
        cached = { signature, node: renderMessage(message) };
        state.messageNodes.set(message.id, cached);
      }
      return cached.node;
    });
    for (const id of state.messageNodes.keys()) if (!retained.has(id)) state.messageNodes.delete(id);
    const sources = snapshot.session?.source_activities || [];
    if (sources.length) {
      const signature = JSON.stringify(sources);
      let panel = state.sourcePanel;
      if (!panel || panel.dataset.signature !== signature) {
      const wasOpen = panel?.open;
      panel = document.createElement("details");
      panel.open = Boolean(wasOpen);
      panel.dataset.signature = signature;
      panel.className = "model-update";
      panel.innerHTML = `<summary>Internet research — ${sources.length} recent source activities</summary>` +
        sources.slice(-20).map(source => `<div><strong>${activityTitle(source)}</strong> — ${escapeHtml(source.status)}<p>${escapeHtml(source.summary || "")}</p><pre style="white-space:pre-wrap">${escapeHtml(source.detail || "")}</pre></div>`).join("");
      state.sourcePanel = panel;
      }
      nodes.push(panel);
    }
    if (snapshot.live) nodes.push(renderLive(snapshot.live));
    if (snapshot.session?.protocol_recovery && !snapshot.live) {
      nodes.push(renderProtocolRecovery(snapshot.session.protocol_recovery));
    }
    (snapshot.computations || []).forEach((calculation) => {
      nodes.push(renderComputation(calculation));
    });
    // Keep unchanged message/plot DOM attached: no image reloads, lost table
    // position, collapsed details, or selection loss on every status update.
    nodes.forEach((node, index) => {
      const current = els.conversation.children[index];
      if (current !== node) els.conversation.insertBefore(node, current || null);
    });
    while (els.conversation.children.length > nodes.length) els.conversation.lastElementChild.remove();
  }

  function renderComputation(calculation) {
    const row = document.createElement("div");
    row.className = "live-message computation-message";
    const progress = calculation.progress || {};
    const telemetry = calculation.telemetry || {};
    const process = telemetry.process || {};
    const status = calculation.cancel_requested ? "Stop requested" :
      telemetry.status === "progress_stale" ? "Calculation active — numerical progress unknown" :
      ["heartbeat_stale", "process_exited"].includes(telemetry.status) ? "Calculation needs attention" : "Numerical campaign running";
    const detail = progress.detail || calculation.detail || "Starting the configured JuTrack workers.";
    const generation = progress.model_independent_status?.generation;
    const generations = progress.generations;
    const evaluated = progress.model_independent_status?.evaluated_in_generation;
    const population = progress.population;
    const elapsed = elapsedLabel(progress.elapsed_s || 0);
    row.innerHTML = `<div class="avatar">B</div><div class="live-card computation-card">
      <div class="live-heading"><div class="live-kicker"><span class="pulse"></span>${escapeHtml(status)}</div><span class="model-badge">JuTrack compute job</span></div>
      <div class="live-title"><strong>${escapeHtml(progress.stage || calculation.tool_id || "Managed calculation")}</strong></div>
      <div class="live-summary"><span>${elapsed} elapsed</span>${generation !== undefined ? `<span>generation ${escapeHtml(generation)}/${escapeHtml(generations || "?")}</span>` : ""}${evaluated !== undefined ? `<span>${escapeHtml(evaluated)}/${escapeHtml(population || "?")} particles</span>` : ""}<span>${escapeHtml(calculation.state.replaceAll("_", " "))}</span></div>
      <div class="model-update"><span>Current status</span><p>${escapeHtml(detail)}</p><p>This calculation is independent of the model turn and will keep running unless you stop it.</p></div>
      <div class="model-update"><span>Observed activity — not a validated result</span>
        <p>${escapeHtml(telemetry.message || "Detailed telemetry is unavailable from this server version.")}</p>
        ${telemetry.program ? `<p>Program: <code>${escapeHtml(telemetry.program)}</code></p>` : ""}
        <p>Runner heartbeat: ${telemetry.heartbeat_age_s != null ? `${elapsedLabel(telemetry.heartbeat_age_s)} ago` : "unknown"}; numerical report: ${telemetry.progress_age_s != null ? `${elapsedLabel(telemetry.progress_age_s)} ago` : "not received"}.</p>
        <p>Process: ${escapeHtml(process.status || "unknown")}${process.pid ? ` (PID ${escapeHtml(process.pid)})` : ""}${process.cpu_seconds != null ? `; CPU time ${elapsedLabel(process.cpu_seconds)}; recent CPU ${process.cpu_percent != null ? `${escapeHtml(process.cpu_percent)}% (100% = one core)` : "sampling"}; memory ${escapeHtml(process.memory_mb)} MB; ${escapeHtml(process.process_count)} processes` : ""}.</p>
        ${(telemetry.files || []).length ? `<ul>${telemetry.files.map(file => `<li>${escapeHtml(file.name)} — ${escapeHtml(file.bytes)} bytes, changed ${elapsedLabel(file.age_s)} ago (unvalidated file)</li>`).join("")}</ul>` : "<p>No nonempty numerical output files observed yet.</p>"}
      </div>
      <div class="computation-actions">${calculation.can_cancel ? `<button type="button" class="stop-computation">Stop calculation…</button>` : `<span>${calculation.cancel_requested ? "Cancellation requested; saved evidence will be retained." : ""}</span>`}</div>
    </div>`;
    const stop = row.querySelector(".stop-computation");
    if (stop) stop.addEventListener("click", async () => {
      if (!confirm("Stop this calculation? Saved artifacts remain, but the current evaluation may be interrupted.")) return;
      stop.disabled = true;
      stop.textContent = "Requesting stop…";
      try {
        await api(`/api/projects/${encodeURIComponent(state.projectId)}/computations/${encodeURIComponent(calculation.id)}/cancel`, { method: "POST", body: "{}" });
        showToast("Stop requested. Saved artifacts are preserved; the current evaluation may be interrupted.");
      } catch (error) {
        stop.disabled = false;
        stop.textContent = "Stop calculation…";
        showToast(error.message);
      }
    });
    return row;
  }

  function renderProtocolRecovery(recovery) {
    const panel = document.createElement("section");
    panel.className = "protocol-recovery";
    panel.setAttribute("aria-label", "Research recovery");
    panel.innerHTML = `<h3>Research paused — choose the next action</h3>
      <p>${escapeHtml(recovery.message)}</p>
      <p>${recovery.checkpoint_available ? "Validated results and generated code are preserved." : "The conversation and generated code are preserved; no validated calculation receipt is available yet."}
      Retrying resumes your latest request. It does not start a new project or reset completed calculations.</p>
      <div class="recovery-actions">
        <button type="button" data-recovery="retry" ${recovery.can_retry ? "" : "disabled"}>Retry from checkpoint</button>
        <button type="button" data-recovery="model">Change model…</button>
        <button type="button" data-recovery="dismiss">Stop here</button>
      </div><p class="recovery-note">Independent numerical jobs keep running unless you stop them with their own controls.</p>`;
    panel.querySelectorAll("[data-recovery]").forEach((button) => {
      button.addEventListener("click", async () => {
        if (button.dataset.recovery === "model") return showModelConfiguration();
        const projectId = state.projectId;
        if (!projectId) return;
        panel.querySelectorAll("button").forEach((item) => { item.disabled = true; });
        try {
          const result = await api(`/api/projects/${encodeURIComponent(projectId)}/protocol-recovery`, {
            method: "POST", body: JSON.stringify({ action: button.dataset.recovery,
              failure_id: recovery.failure_id, request_id: recovery.request_id })
          });
          if (projectId !== state.projectId) return;
          showToast(result.detail);
          applySnapshot(await api(`/api/projects/${encodeURIComponent(projectId)}`));
        } catch (error) {
          showToast(error.message);
          if (projectId === state.projectId) {
            panel.querySelectorAll("button").forEach((item) => { item.disabled = item.dataset.recovery === "retry" && !recovery.can_retry; });
          }
        }
      });
    });
    return panel;
  }

  function showModelConfiguration() {
    let dialog = document.querySelector("#model-configuration");
    if (dialog) dialog.remove();
    dialog = document.createElement("dialog");
    dialog.id = "model-configuration";
    dialog.className = "model-configuration";
    dialog.innerHTML = `<h2>Models &amp; account</h2>
      <p>ChatGPT subscription uses official Codex sign-in. IHEP and OpenAI API billing remain separate.</p>
      <label>Backend <select id="backend-profile"></select></label>
      <label>Model <input id="api-model" autocomplete="off"><select id="subscription-model" hidden></select></label>
      <button id="apply-model" type="button">Use for next turn</button>
      <p id="model-settings-status" role="status">Loading configuration…</p>
      <hr><h3>Codex-managed ChatGPT account</h3>
      <p>Sign-in uses the local Codex account, shared with Codex CLI. Signing out affects that account too.</p>
      <div id="codex-account-status">Checking account…</div><div id="codex-login-link"></div>
      <button id="codex-login" type="button">Sign in with ChatGPT</button>
      <button id="codex-device" type="button">Use device code</button>
      <button id="codex-cancel" type="button">Cancel sign-in</button>
      <button id="codex-logout" type="button">Sign out</button>
      <button id="codex-refresh" type="button">Refresh status</button>
      <p>Models and quota come from Codex; availability depends on your account. No cookies or tokens are copied into Brilliance.</p>
      <form method="dialog"><button type="submit">Close</button></form>`;
    document.body.appendChild(dialog);
    dialog.showModal();
    let settings = null, account = null, timer = null;
    const within = (id) => dialog.querySelector(id);
    const message = (value) => { within("#model-settings-status").textContent = value; };
    const chooseProfile = () => {
      const profile = settings?.profiles.find(p => p.id === within("#backend-profile").value);
      const subscription = profile?.provider === "chatgpt";
      within("#api-model").hidden = subscription;
      within("#subscription-model").hidden = !subscription;
      within("#api-model").value = profile?.model || "";
      within("#subscription-model").value = profile?.model || "";
    };
    const displayLogin = (login) => {
      const target = within("#codex-login-link");
      target.replaceChildren();
      if (login?.status === "pending" && /^https:\/\//i.test(login.url || "")) {
        const link = document.createElement("a");
        link.href = login.url; link.target = "_blank"; link.rel = "noopener noreferrer";
        link.textContent = "Open official Codex sign-in";
        target.append(link);
        if (login.user_code) { const code = document.createElement("p"); code.textContent = `Device code: ${login.user_code}`; target.append(code); }
      } else if (login?.message) target.textContent = login.message;
    };
    const refresh = async () => {
      clearTimeout(timer);
      account = await api("/api/account/codex");
      if (!dialog.open) return;
      const info = account.account || {};
      const windows = account.usage?.rateLimitsByLimitId || (account.usage?.rateLimits ? {codex: account.usage.rateLimits} : {});
      const quota = Object.entries(windows).flatMap(([name, bucket]) => ["primary", "secondary"].filter(k => bucket?.[k]).map(k => {
        const w = bucket[k];
        return `${name} ${w.windowDurationMins ?? "?"}-minute window: ${w.usedPercent ?? "unknown"}% used; reset ${w.resetsAt ? new Date(w.resetsAt * 1000).toLocaleString() : "unknown"}`;
      }));
      within("#codex-account-status").textContent = [
        info.authenticated ? `Signed in: ${info.email || "ChatGPT account"} (${info.plan || "plan unavailable"})` : "Not signed in with ChatGPT",
        account.runtime_versions ? `Codex runtime ${account.runtime_versions.runtime}; SDK ${account.runtime_versions.sdk}` : "",
        account.error || "", ...quota, ...(account.warnings || []),
        !quota.length ? "Quota unavailable; this does not mean unlimited usage." : "",
      ].filter(Boolean).join(" · ");
      within("#subscription-model").innerHTML = `<option value="">Codex-reported default</option>` + (account.models || []).map(m =>
        `<option value="${escapeHtml(m.model)}">${escapeHtml(m.displayName || m.model)}</option>`).join("");
      chooseProfile(); displayLogin(account.login);
      if (account.login?.status === "pending") timer = setTimeout(() => refresh().catch(e => message(e.message)), 3000);
    };
    within("#backend-profile").onchange = chooseProfile;
    within("#apply-model").onclick = async () => {
      try {
        const profile = settings.profiles.find(p => p.id === within("#backend-profile").value);
        settings = await api("/api/settings/models", {method: "POST", body: JSON.stringify({
          profile: profile.id, model: within(profile.provider === "chatgpt" ? "#subscription-model" : "#api-model").value})});
        message("Saved. The next model turn uses this selection. Numerical jobs and project evidence are unchanged.");
      } catch (e) { message(e.message); }
    };
    const login = async (method) => {
      try { displayLogin(await api("/api/account/codex/login", {method: "POST", body: JSON.stringify({method})}));
        timer = setTimeout(() => refresh().catch(e => message(e.message)), 3000);
      } catch (e) { message(e.message); }
    };
    within("#codex-login").onclick = () => login("browser");
    within("#codex-device").onclick = () => login("device");
    within("#codex-refresh").onclick = () => refresh().catch(e => message(e.message));
    for (const action of ["cancel", "logout"]) within(`#codex-${action}`).onclick = async () => {
      if (action === "logout" && !confirm("Sign out of the shared Codex ChatGPT account?")) return;
      try { await api(`/api/account/codex/${action}`, {method: "POST"}); await refresh(); }
      catch (e) { message(e.message); }
    };
    dialog.addEventListener("close", () => clearTimeout(timer));
    api("/api/settings/models").then(data => {
      settings = data;
      within("#backend-profile").innerHTML = data.profiles.map(p => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name)}</option>`).join("");
      within("#backend-profile").value = data.selected;
      chooseProfile(); message(`Configuration: ${data.source}. Credentials stay in .env.`);
      return refresh();
    }).catch(e => message(e.message));
  }

  $("#model-settings")?.addEventListener("click", showModelConfiguration);

  function renderMessage(message) {
    const article = document.createElement("article");
    article.className = `message ${message.role}`;
    if (message.meta?.presentation_role === "technical_details") article.classList.add("technical-message");
    if (state.newMessageIds.has(message.id)) article.classList.add("message-new");
    article.dataset.messageId = message.id;
    if (message.role === "assistant") {
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = "B";
      article.appendChild(avatar);
    }
    const body = document.createElement("div");
    body.className = "message-body";
    if (message.meta?.stage_id || message.meta?.candidate_id) {
      body.appendChild(renderScientificContext(message.meta));
    }
    const content = document.createElement("div");
    content.className = "message-content";
    content.innerHTML = markdown(message.content);
    if (message.content) body.appendChild(content);
    if (message.attachments?.length) {
      const attachments = document.createElement("div");
      attachments.className = "message-attachments";
      deduplicateAttachments(message.attachments).forEach(
        (attachment) => attachments.appendChild(renderAttachment(attachment))
      );
      body.appendChild(attachments);
    }
    const time = document.createElement("div");
    time.className = "message-time";
    time.textContent = timeLabel(message.timestamp);
    body.appendChild(time);
    if (message.meta?.conversation_checkpoint) {
      const restore = document.createElement("button");
      restore.className = "checkpoint-branch";
      restore.textContent = "Branch from here";
      restore.title = "Restore saved files and conversation into a new project; the original and its jobs remain unchanged.";
      restore.addEventListener("click", async () => {
        if (!window.confirm("Create a new project from the saved state at this message? The original project and its running jobs will remain unchanged. No calculation will restart automatically.")) return;
        restore.disabled = true;
        try {
          const result = await api(`/api/projects/${encodeURIComponent(state.projectId)}/checkpoints/${encodeURIComponent(message.meta.conversation_checkpoint)}/branch`, { method: "POST" });
          await openProject(result.project.id);
        } catch (error) { showToast(error.message); }
        finally { restore.disabled = false; }
      });
      body.appendChild(restore);
    }
    article.appendChild(body);
    return article;
  }

  function renderLive(live) {
    const row = document.createElement("div");
    row.className = "live-message";
    const progress = live.progress || {};
    const activities = (progress.milestones?.length
      ? progress.milestones
      : (live.activities || []).slice(-6))
      .slice(-4);
    const updates = (progress.updates || []).slice(-1);
    const findings = (progress.latest_results || []).slice(-3);
    const errors = (progress.latest_errors || []).slice(-2);
    const sourceNotices = (progress.source_notices || []).slice(-2);
    const narrative = live.partial_response || progress.narrative || "";
    const pending = Number(live.pending_count || 0);
    const backend = live.backend || {};
    const currentModel = modelLabel(backend);
    const researchNetwork = live.research_network || {};
    const queued = live.state === "queued";
    const title = queued ? live.label : (progress.title || live.label);
    const increment = Number(progress.increment || 1);
    const maxIncrements = Number(progress.max_increments || 1);
    const completed = Number(progress.completed || 0);
    const repairs = Number(progress.failed || 0);
    row.innerHTML = `<div class="avatar">B</div><div class="live-card">
      <div class="live-heading"><div class="live-kicker"><span class="pulse"></span>Research in progress</div><span class="model-badge">${escapeHtml(currentModel)} · ${networkLabel(researchNetwork)}</span></div>
      <div class="live-title"><strong>${escapeHtml(title)}</strong></div>
      <div class="live-summary"><span data-live-elapsed>${elapsedLabel(progress.elapsed_seconds)} elapsed</span><span>You can send a message while I work</span></div>
      ${narrative ? `<div class="model-update message-content">${markdown(narrative)}</div>` : ""}
      ${updates.length ? `<div class="research-log"><div class="live-section-title">Latest research update</div>${updates.map((update) =>
        `<div class="research-update ${escapeHtml(update.kind || "finding")}"><span>${escapeHtml(update.kind || "finding")}</span><div><strong>${escapeHtml(update.stage)}</strong>${update.detail ? `<p>${escapeHtml(update.detail)}</p>` : ""}</div></div>`
      ).join("")}</div>` : ""}
      ${findings.length ? `<div class="finding-list"><div class="live-section-title">Latest computed results</div>${findings.map((finding) =>
        `<div class="finding-row"><span class="activity-status completed"></span><div><strong>${escapeHtml(finding.title)}</strong><p>${escapeHtml(finding.summary)}</p></div></div>`
      ).join("")}</div>` : ""}
      ${errors.length ? `<div class="finding-list error-list"><div class="live-section-title">What needs repair</div>${errors.map((error) =>
        `<div class="finding-row"><span class="activity-status failed"></span><div><strong>${escapeHtml(error.title)}</strong><p>${escapeHtml(error.summary)}</p></div></div>`
      ).join("")}</div>` : ""}
      ${sourceNotices.length ? `<div class="finding-list"><div class="live-section-title">Source unavailable — research can continue</div>${sourceNotices.map((notice) =>
        `<div class="finding-row"><div><strong>${escapeHtml(notice.title)}</strong><p>${escapeHtml(notice.summary)}</p></div></div>`
      ).join("")}</div>` : ""}
      ${activities.length ? `<div class="activity-list">${activities.map((activity) =>
        `<div class="activity-row"><span class="activity-status ${escapeHtml(activity.status || "running")}"></span><div><span>${activityTitle(activity)}</span>${activity.summary ? `<p>${escapeHtml(activity.summary)}</p>` : ""}</div><small>${escapeHtml((activity.status || "running").replaceAll("_", " "))}</small></div>`
      ).join("")}</div>` : ""}
      ${pending ? `<div class="pending-note">${pending} follow-up message${pending === 1 ? "" : "s"} queued</div>` : ""}
    </div>`;
    row.querySelectorAll(".error-list, .activity-list, .finding-list").forEach((section) => {
      const sourceNotice = section.querySelector(".live-section-title")?.textContent.startsWith("Source unavailable");
      if (!sourceNotice && !section.matches(".error-list, .activity-list")) return;
      const details = document.createElement("details");
      details.className = "technical-details";
      const summary = document.createElement("summary");
      summary.textContent = sourceNotice ? "Source-access notices" : section.matches(".error-list") ? "Execution diagnostics" : "Tool activity";
      const key = sourceNotice ? "sources" : section.matches(".error-list") ? "errors" : "tools";
      details.dataset.diagnostic = key;
      details.open = document.querySelector(`.live-message details[data-diagnostic="${key}"]`)?.open || false;
      section.replaceWith(details);
      details.append(summary, section);
    });
    return row;
  }

  function renderAttachment(attachment) {
    if (attachment.type === "technical_details") {
      const details = document.createElement("details");
      details.className = "technical-details";
      const rows = attachment.rows || [];
      const heading = document.createElement("summary");
      heading.textContent = `${attachment.title || "Technical details"} (${rows.length})`;
      details.appendChild(heading);
      rows.forEach((row) => {
        const entry = document.createElement("div");
        entry.dataset.messageId = row.id || "";
        const time = document.createElement("small");
        time.textContent = timeLabel(row.timestamp);
        const text = document.createElement("pre");
        text.textContent = row.text || "";
        entry.append(time, text);
        details.appendChild(entry);
      });
      return details;
    }
    if (attachment.type === "tool") return renderTool(attachment);
    if (attachment.type === "metrics") return renderMetrics(attachment);
    if (attachment.type === "evidence_summary") return renderEvidenceSummary(attachment);
    if (attachment.type === "alternative_results") {
      const details = document.createElement("details");
      details.className = "alternative-results";
      const heading = document.createElement("summary");
      heading.textContent = `${attachment.title || "Other calculations"} (${(attachment.rows || []).length})`;
      details.appendChild(heading);
      (attachment.rows || []).forEach((row) => {
        const entry = document.createElement("div");
        entry.className = "alternative-result";
        const identity = document.createElement("div");
        identity.className = "scientific-context";
        identity.textContent = [row.run_id, row.status, "Not selected for this answer"].filter(Boolean).join(" · ");
        const text = document.createElement("p");
        text.textContent = row.summary || "";
        entry.append(identity, text);
        details.appendChild(entry);
      });
      return details;
    }
    if (attachment.type === "evidence_reference") {
      const link = document.createElement("button");
      link.type = "button";
      link.className = "evidence-reference";
      link.textContent = `Previously shown: ${attachment.title || "Evidence"}`;
      link.addEventListener("click", () => {
        const target = [...document.querySelectorAll("[data-message-id]")].find(
          (item) => item.dataset.messageId === attachment.message_id
        );
        target?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
      return link;
    }
    if (attachment.type === "activity") {
      const details = document.createElement("details");
      details.className = "tool-card activity-history";
      const rows = attachment.rows || [];
      details.innerHTML = `<summary><span class="tool-dot"></span><strong>${escapeHtml(attachment.title || "Research activity")}</strong><span>${rows.length} steps</span></summary>
        <div class="tool-body"><div class="activity-list">${rows.map((activity) =>
          `<div class="activity-row"><span class="activity-status ${escapeHtml(activity.status || "completed")}"></span><div><span>${activityTitle(activity)}</span>${activity.summary ? `<p>${escapeHtml(activity.summary)}</p>` : ""}</div><small>${escapeHtml((activity.status || "completed").replaceAll("_", " "))}</small></div>`
        ).join("")}</div></div>`;
      return details;
    }
    if (attachment.type === "artifact") return renderArtifactShell(attachment);
    if (attachment.type === "image") {
      const figure = document.createElement("figure");
      figure.className = "generated-figure";
      const image = document.createElement("img");
      image.src = attachment.url;
      image.alt = attachment.title || "Research plot";
      image.loading = "lazy";
      const caption = document.createElement("figcaption");
      caption.textContent = [attachment.title || "Research plot", attachment.lattice_id].filter(Boolean).join(" · ");
      figure.append(image, caption);
      return figure;
    }
    if (attachment.type === "download") {
      const link = document.createElement("a");
      link.className = "download-card";
      link.href = attachment.url;
      link.innerHTML = `↓ <span>${escapeHtml(attachment.title)}</span>`;
      return link;
    }
    return document.createElement("span");
  }

  function renderTool(tool) {
    const details = document.createElement("details");
    details.className = "tool-card";
    details.innerHTML = `<summary><span class="tool-dot"></span><strong>${escapeHtml(tool.title)}</strong><span>${escapeHtml(tool.status)}</span></summary>
      <div class="tool-body"><dl>
        <dt>Numerical source</dt><dd>${escapeHtml(tool.source || "registered JuTrack service")}</dd>
        <dt>Evidence gate</dt><dd>${escapeHtml(tool.validator || "registered deterministic validator")}</dd>
        <dt>Evidence ID</dt><dd>${escapeHtml(tool.evidence_id || "n/a")}</dd>
      </dl></div>`;
    return details;
  }

  function renderMetrics(attachment) {
    const card = document.createElement("div");
    card.className = "metrics-card";
    const rows = attachment.rows || [];
    card.innerHTML = `<div class="card-title">${escapeHtml(attachment.title)}</div><div class="metrics-grid">${rows.map((row) =>
      `<div class="metric"><span class="label">${escapeHtml(row.quantity)}</span><span class="value">${escapeHtml(row.value)}</span><span class="unit">${escapeHtml(row.unit)}</span></div>`
    ).join("")}</div>`;
    return card;
  }

  function renderScientificContext(stage = {}, latticeId = "") {
    const context = document.createElement("div");
    context.className = "scientific-context";
    const labels = [stage.stage_id, stage.hypothesis_id && `Hypothesis: ${stage.hypothesis_id}`,
      stage.outcome && stage.outcome.replaceAll("_", " "),
      (latticeId || stage.candidate_id) && `Candidate: ${latticeId || stage.candidate_id}`,
      stage.parent_candidate_id && `Parent: ${stage.parent_candidate_id}`].filter(Boolean);
    context.textContent = labels.join(" · ");
    return context;
  }

  function renderEvidenceSummary(attachment) {
    const card = document.createElement("section");
    card.className = "evidence-summary-card";
    const title = document.createElement("div");
    title.className = "card-title";
    title.textContent = attachment.title || "Scientific stage evidence";
    card.appendChild(title);
    if (attachment.presentation_role === "selected_evidence") {
      const selected = document.createElement("div");
      selected.className = "selected-evidence-label";
      const working = attachment.working_lattice_action === "advance";
      const selectionRequested = attachment.candidate_handoff?.disposition === "working_selection_requested";
      selected.textContent = "Evidence for this answer" + (attachment.candidate_lattice
        ? working ? " · selected as provisional working candidate by this calculation"
          : !attachment.working_lattice_action && selectionRequested
            ? " · working selection requested in the saved result"
            : " · archived candidate, not a working-lattice selection"
        : "");
      card.appendChild(selected);
    }
    if (attachment.history_rehydrated) {
      const restored = document.createElement("p");
      restored.className = "evidence-summary-text";
      restored.textContent = attachment.history_rehydrated;
      card.appendChild(restored);
    }
    const stage = attachment.scientific_stage || {};
    if (attachment.candidate_lattice && attachment.lattice_id) {
      const identity = document.createElement("p");
      identity.className = "evidence-summary-text";
      const active = state.snapshot?.project?.active_lattice_id;
      identity.textContent = `Displayed candidate: ${attachment.lattice_id}` +
        (active ? ` · Current working lattice: ${active}` : "") +
        (active && active !== attachment.lattice_id ? " · Separate trial; does not replace the working lattice." : "");
      card.appendChild(identity);
    }
    if (attachment.outcome_presentation?.evidence_status) {
      const status = document.createElement("p");
      status.className = "evidence-summary-text";
      status.textContent = `Evidence status: ${attachment.outcome_presentation.evidence_status.replaceAll("_", " ")}`;
      card.appendChild(status);
    }
    if (attachment.lattice_id || Object.keys(stage).length) {
      card.appendChild(renderScientificContext(stage, attachment.lattice_id));
    }
    if (attachment.run_id) {
      const receipt = document.createElement("div");
      receipt.className = "evidence-receipt";
      receipt.textContent = `Calculation: ${attachment.run_id}`;
      card.appendChild(receipt);
    }
    if (attachment.summary) {
      const summary = document.createElement("p");
      summary.className = "evidence-summary-text";
      summary.textContent = attachment.summary;
      card.appendChild(summary);
    }
    if (attachment.program_stdout) {
      const output = document.createElement("details");
      const label = document.createElement("summary");
      label.textContent = "Program output";
      const text = document.createElement("pre");
      text.textContent = attachment.program_stdout;
      output.append(label, text);
      card.appendChild(output);
    }
    if (attachment.scientific_passed !== null && attachment.scientific_passed !== undefined) {
      const gate = document.createElement("p");
      gate.className = "evidence-summary-text";
      gate.textContent = attachment.scientific_passed
        ? "The research program reports that its checks passed. Brilliance does not independently certify the design."
        : "The research program reports failed checks; see the measurements and explanation.";
      card.appendChild(gate);
    }
    if (attachment.steps?.length) {
      const methods = document.createElement("details");
      methods.className = "evidence-methods";
      const heading = document.createElement("summary");
      heading.textContent = "Calculation methods and provenance";
      methods.appendChild(heading);
      const list = document.createElement("ol");
      list.className = "evidence-method-steps";
      attachment.steps.forEach((step) => {
        const item = document.createElement("li");
        item.textContent = step;
        list.appendChild(item);
      });
      methods.appendChild(list);
      card.appendChild(methods);
    }
    const evidenceTables = attachment.tables || [];
    const isAutomaticQuantities = entry => entry.kind === "automatic_quantities"
      || (!entry.kind && entry.title === "Computed quantities"); // Legacy platform-generated table.
    const hasCuratedTables = evidenceTables.some(entry => !isAutomaticQuantities(entry) && entry.kind !== "candidate_evaluation");
    evidenceTables.forEach((entry) => {
      const rows = entry.rows || [];
      // Large diagnostic inventories remain available without pushing the
      // selected candidate's figures below pages of auxiliary rows. A typed
      // authoritative receipt stays accessible in full; a curated scientific
      // summary takes precedence over repeating every verified quantity.
      const verifiedDetails = entry.kind === "candidate_evaluation" && hasCuratedTables;
      const automaticDetails = isAutomaticQuantities(entry) && evidenceTables.some(table => !isAutomaticQuantities(table));
      if (verifiedDetails || automaticDetails || (rows.length > 16 && entry.kind !== "candidate_evaluation")) {
        const details = document.createElement("details");
        details.className = "detailed-evidence-table" + (verifiedDetails ? " verified-evidence-details" : automaticDetails ? " automatic-quantity-details" : "");
        const heading = document.createElement("summary");
        heading.textContent = verifiedDetails ? `Verified numerical details (${rows.length} quantities)`
          : automaticDetails ? `Additional recorded quantities (${rows.length})`
            : `${entry.title || "Detailed evidence"} · ${rows.length} rows`;
        details.appendChild(heading);
        if (entry.description) {
          const description = document.createElement("p");
          description.textContent = entry.description;
          details.appendChild(description);
        }
        details.appendChild(makeTable(rows));
        card.appendChild(details);
        return;
      }
      const heading = document.createElement("div");
      heading.className = "evidence-table-title";
      heading.textContent = entry.title || "Evidence";
      card.appendChild(heading);
      if (entry.description) {
        const description = document.createElement("p");
        description.className = "evidence-summary-text";
        description.textContent = entry.description;
        card.appendChild(description);
      }
      card.appendChild(makeTable(rows));
    });
    for (const [key, label] of [["interpretation", "Interpretation"], ["next_decision", "Next scientific decision"]]) {
      if (!stage[key]) continue;
      const text = document.createElement("p");
      text.className = "evidence-summary-text";
      text.textContent = `${label}: ${stage[key]}`;
      card.appendChild(text);
    }
    if (attachment.pending_acceptance?.length) {
      const heading = document.createElement("div");
      heading.className = "evidence-table-title";
      heading.textContent = "Pending final acceptance";
      card.appendChild(heading);
      const list = document.createElement("ul");
      list.className = "coded-notes";
      attachment.pending_acceptance.forEach((value) => {
        const item = document.createElement("li");
        item.textContent = value;
        list.appendChild(item);
      });
      card.appendChild(list);
    }
    if (attachment.limitations?.length) {
      const heading = document.createElement("div");
      heading.className = "evidence-table-title";
      heading.textContent = "Limitations";
      card.appendChild(heading);
      const list = document.createElement("ul");
      list.className = "coded-notes";
      attachment.limitations.forEach((value) => {
        const item = document.createElement("li");
        item.textContent = value;
        list.appendChild(item);
      });
      card.appendChild(list);
    }
    if (attachment.evidence_sources?.length > 1) {
      const sources = document.createElement("details");
      sources.className = "evidence-sources";
      const heading = document.createElement("summary");
      heading.textContent = `${attachment.evidence_sources.length} calculation receipts — identical presentation shown once`;
      sources.appendChild(heading);
      const list = document.createElement("ul");
      attachment.evidence_sources.forEach((source) => {
        const item = document.createElement("li");
        item.textContent = [source.run_id, source.result_path].filter(Boolean).join(" · ");
        list.appendChild(item);
      });
      sources.appendChild(list);
      card.appendChild(sources);
    }
    return card;
  }

  function renderArtifactShell(attachment) {
    const details = document.createElement("details");
    details.className = "artifact-card";
    details.dataset.artifactId = attachment.artifact_id;
    const shouldOpen = attachment.expanded || state.openArtifacts.has(attachment.artifact_id);
    details.open = shouldOpen;
    details.innerHTML = `<summary><span>⌁</span><strong>${escapeHtml(attachment.title)}</strong></summary><div class="artifact-body"><div class="artifact-loading"><span class="spinner"></span>Loading accepted JuTrack data…</div></div>`;
    details.addEventListener("toggle", () => {
      if (details.open) {
        state.openArtifacts.add(attachment.artifact_id);
        loadArtifact(details, attachment);
      } else state.openArtifacts.delete(attachment.artifact_id);
    });
    if (shouldOpen) requestAnimationFrame(() => loadArtifact(details, attachment));
    return details;
  }

  async function loadArtifact(details, attachment) {
    const body = details.querySelector(".artifact-body");
    if (body.dataset.loaded === "true") return;
    try {
      let data = state.artifactCache.get(attachment.url);
      if (!data) {
        data = await api(attachment.url);
        state.artifactCache.set(attachment.url, data);
      }
      body.replaceChildren(renderArtifact(data));
      body.dataset.loaded = "true";
    } catch (error) {
      body.innerHTML = `<div class="artifact-loading">Could not load this view: ${escapeHtml(error.message)}</div>`;
    }
  }

  function renderArtifact(data) {
    const wrap = document.createElement("div");
    if (data.kind === "lattice") wrap.appendChild(ringChart(data));
    else if (data.kind === "lattice_block") wrap.appendChild(latticeChart(data));
    else if (data.kind === "optics") wrap.appendChild(opticsChart(data));
    else if (data.kind === "radiation") wrap.appendChild(radiationView(data));
    else if (data.kind === "chromaticity") wrap.appendChild(chromaticityChart(data));
    else if (data.kind === "dynamic_aperture") wrap.appendChild(apertureChart(data));
    else if (data.kind === "robustness") wrap.appendChild(robustnessChart(data));
    else if (data.kind === "comparison") wrap.appendChild(comparisonTable(data));
    else if (data.kind === "research_plots") wrap.appendChild(researchPlots(data));
    else if (data.kind === "particle_tracking") wrap.appendChild(particleTrackingCharts(data));
    else if (data.kind === "population_optimization") wrap.appendChild(populationOptimizationView(data));
    else if (data.kind === "coded_experiment") wrap.appendChild(codedExperimentView(data));
    else if (data.kind === "dataset_catalog") wrap.appendChild(datasetCatalogView(data));
    else if (data.kind === "data_view") wrap.appendChild(dataView(data));
    else wrap.textContent = "No visual renderer is registered for this result.";
    return wrap;
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  let chartSequence = 0;
  function svg(width = 820, height = 330) {
    const node = document.createElementNS(SVG_NS, "svg");
    node.setAttribute("viewBox", `0 0 ${width} ${height}`);
    node.setAttribute("class", "chart");
    node.setAttribute("role", "img");
    return node;
  }
  function add(parent, tag, attrs = {}, text = "") {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== "") node.textContent = text;
    parent.appendChild(node);
    return node;
  }
  function domain(values, pad = .05) {
    const finite = values.map(Number).filter(Number.isFinite);
    if (!finite.length) return [0, 1];
    let min = Math.min(...finite), max = Math.max(...finite);
    if (min === max) { min -= 1; max += 1; }
    const gap = (max - min) * pad;
    return [min - gap, max + gap];
  }
  function scale([d0, d1], [r0, r1]) { return (value) => r0 + (Number(value) - d0) * (r1 - r0) / (d1 - d0); }
  function axes(node, xDomain, yDomain, width, height, labels = {}) {
    const m = { left: 52, right: 20, top: 17, bottom: 39 };
    const sx = scale(xDomain, [m.left, width - m.right]);
    const sy = scale(yDomain, [height - m.bottom, m.top]);
    for (let i = 0; i <= 4; i++) {
      const y = m.top + i * (height - m.top - m.bottom) / 4;
      add(node, "line", { x1: m.left, x2: width - m.right, y1: y, y2: y, class: "grid" });
      const value = yDomain[1] - i * (yDomain[1] - yDomain[0]) / 4;
      add(node, "text", { x: m.left - 7, y: y + 3, "text-anchor": "end" }, formatValue(value, 3));
    }
    for (let i = 0; i <= 4; i++) {
      const x = m.left + i * (width - m.left - m.right) / 4;
      const value = xDomain[0] + i * (xDomain[1] - xDomain[0]) / 4;
      add(node, "text", { x, y: height - 17, "text-anchor": "middle" }, formatValue(value, 3));
    }
    add(node, "line", { x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom, class: "axis" });
    add(node, "line", { x1: m.left, x2: m.left, y1: m.top, y2: height - m.bottom, class: "axis" });
    if (labels.x) add(node, "text", { x: (m.left + width - m.right) / 2, y: height - 2, "text-anchor": "middle" }, labels.x);
    if (labels.y) add(node, "text", { x: 11, y: height / 2, transform: `rotate(-90 11 ${height/2})`, "text-anchor": "middle" }, labels.y);
    return { sx, sy, m };
  }
  function linePath(rows, x, y, sx, sy) {
    return rows.map((row, index) => `${index ? "L" : "M"}${sx(row[x]).toFixed(2)},${sy(row[y]).toFixed(2)}`).join(" ");
  }
  function chartContainer(node, legendRows = [], meta = {}) {
    const wrap = document.createElement("div"); wrap.className = "chart-wrap";
    if (legendRows.length) {
      const legend = document.createElement("div"); legend.className = "legend";
      legend.innerHTML = legendRows.map((item) => `<span><i style="background:${item.color}"></i>${escapeHtml(item.label)}</span>`).join("");
      wrap.appendChild(legend);
    }
    wrap.appendChild(node);
    const entries = Object.entries(meta || {}).filter(([, value]) => value !== null && value !== undefined && typeof value !== "object");
    if (entries.length) {
      const line = document.createElement("div"); line.className = "chart-meta";
      line.innerHTML = entries.map(([key, value]) => `<span>${escapeHtml(key.replaceAll("_", " "))}: <b>${escapeHtml(formatValue(value))}</b></span>`).join("");
      wrap.appendChild(line);
    }
    return wrap;
  }

  function ringChart(data) {
    const rows = data.rows || [], node = svg(820, 450);
    const xd = domain(rows.map((r) => r.x_m), .08), yd = domain(rows.map((r) => r.y_m), .08);
    const sx = scale(xd, [25, 795]), sy = scale(yd, [425, 25]);
    const grouped = Map.groupBy ? Map.groupBy(rows, (r) => r.segment) : rows.reduce((map, row) => (map.set(row.segment, [...(map.get(row.segment) || []), row]), map), new Map());
    grouped.forEach((points) => add(node, "path", {
      d: linePath(points, "x_m", "y_m", sx, sy), fill: "none", stroke: points[0]?.color || "#888",
      "stroke-width": points[0]?.family === "Drift" ? 1.1 : 2.5, "stroke-linecap": "round",
    }));
    add(node, "text", { x: 410, y: 444, "text-anchor": "middle" }, "Physical floor plan (m)");
    const families = [...new Map(rows.map((row) => [row.family, row.color])).entries()].map(([label, color]) => ({ label, color }));
    return chartContainer(node, families, data.meta);
  }

  function latticeChart(data) {
    const rows = data.rows || [], node = svg(820, 230);
    const xd = domain(rows.flatMap((r) => [r.s_start_m, r.s_end_m]), 0);
    const sx = scale(xd, [45, 800]), sy = scale([-1, 1], [175, 35]);
    add(node, "line", { x1: 45, x2: 800, y1: sy(0), y2: sy(0), class: "axis" });
    rows.forEach((row) => add(node, "rect", {
      x: sx(row.s_start_m), width: Math.max(1, sx(row.s_end_m) - sx(row.s_start_m)),
      y: Math.min(sy(row.y0), sy(row.y1)), height: Math.max(2, Math.abs(sy(row.y1) - sy(row.y0))),
      fill: row.color, opacity: row.family === "Drift" ? .35 : .9,
    }));
    add(node, "text", { x: 420, y: 211, "text-anchor": "middle" }, "Longitudinal position s (m)");
    const families = [...new Map(rows.map((row) => [row.family, row.color])).entries()].map(([label, color]) => ({ label, color }));
    return chartContainer(node, families, data.meta);
  }

  function opticsChart(data) {
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const latticeRows = Array.isArray(data.lattice_rows) ? data.lattice_rows : [];
    const betaRows = rows.filter((row) => row.group === "beta");
    const dispersionRows = rows.filter((row) => row.group === "dispersion");
    const suppliedDomain = Array.isArray(data.s_domain_m) ? data.s_domain_m.map(Number) : [];
    const xValues = [
      ...rows.map((row) => row.s_m),
      ...latticeRows.flatMap((row) => [row.s_start_m, row.s_end_m]),
    ];
    const xDomain = suppliedDomain.length === 2
      && suppliedDomain.every(Number.isFinite)
      && suppliedDomain[1] > suppliedDomain[0]
      ? suppliedDomain : domain(xValues, 0);

    const width = 820, height = 440;
    const margins = { left: 58, right: 62, top: 18 };
    const mainBottom = 302, latticeTop = 330, latticeBottom = 382;
    const plotRight = width - margins.right;
    const sx = scale(xDomain, [margins.left, plotRight]);
    const betaMax = Math.max(1, ...betaRows.map((row) => Number(row.value)).filter(Number.isFinite));
    const betaDomain = [0, betaMax * 1.06];
    const dispersionDomain = domain([
      0, ...dispersionRows.map((row) => row.value),
    ], .08);
    const syBeta = scale(betaDomain, [mainBottom, margins.top]);
    const syDispersion = scale(dispersionDomain, [mainBottom, margins.top]);
    const syLattice = scale([-1, 1], [latticeBottom, latticeTop]);
    const node = svg(width, height);

    for (let index = 0; index <= 5; index++) {
      const fraction = index / 5;
      const y = margins.top + fraction * (mainBottom - margins.top);
      add(node, "line", {
        x1: margins.left, x2: plotRight, y1: y, y2: y, class: "grid",
      });
      add(node, "text", {
        x: margins.left - 7, y: y + 3, "text-anchor": "end",
      }, formatValue(betaDomain[1] * (1 - fraction), 3));
      const dispersionValue = dispersionDomain[1]
        - fraction * (dispersionDomain[1] - dispersionDomain[0]);
      add(node, "text", {
        x: plotRight + 7, y: y + 3, "text-anchor": "start",
      }, formatValue(dispersionValue, 3));
    }
    for (let index = 0; index <= 5; index++) {
      const fraction = index / 5;
      const x = margins.left + fraction * (plotRight - margins.left);
      const value = xDomain[0] + fraction * (xDomain[1] - xDomain[0]);
      add(node, "line", {
        x1: x, x2: x, y1: margins.top, y2: latticeBottom, class: "grid",
      });
      add(node, "text", {
        x, y: 407, "text-anchor": "middle",
      }, formatValue(value, 4));
    }
    add(node, "line", {
      x1: margins.left, x2: margins.left, y1: margins.top, y2: mainBottom, class: "axis",
    });
    add(node, "line", {
      x1: plotRight, x2: plotRight, y1: margins.top, y2: mainBottom, class: "axis",
    });
    add(node, "line", {
      x1: margins.left, x2: plotRight, y1: mainBottom, y2: mainBottom, class: "axis",
    });
    add(node, "text", {
      x: 13, y: (margins.top + mainBottom) / 2,
      transform: `rotate(-90 13 ${(margins.top + mainBottom) / 2})`,
      "text-anchor": "middle",
    }, "β (m)");
    add(node, "text", {
      x: width - 9, y: (margins.top + mainBottom) / 2,
      transform: `rotate(90 ${width - 9} ${(margins.top + mainBottom) / 2})`,
      "text-anchor": "middle",
    }, "Dx (m)");

    const clipId = `optics-clip-${++chartSequence}`;
    const definitions = add(node, "defs");
    const clip = add(definitions, "clipPath", { id: clipId });
    add(clip, "rect", {
      x: margins.left, y: margins.top,
      width: plotRight - margins.left, height: mainBottom - margins.top,
    });
    [
      { quantity: "βx", color: "#b3472e", rows: betaRows, scale: syBeta },
      { quantity: "βy", color: "#2f6f9f", rows: betaRows, scale: syBeta },
      { quantity: "Dx", color: "#2f7655", rows: dispersionRows, scale: syDispersion },
    ].forEach((series) => {
      const points = series.rows
        .filter((row) => row.quantity === series.quantity)
        .sort((left, right) => Number(left.s_m) - Number(right.s_m));
      if (!points.length) return;
      add(node, "path", {
        d: linePath(points, "s_m", "value", sx, series.scale),
        fill: "none", stroke: series.color, "stroke-width": 1.9,
        "clip-path": `url(#${clipId})`,
      });
    });

    add(node, "line", {
      x1: margins.left, x2: plotRight,
      y1: syLattice(0), y2: syLattice(0), class: "axis",
    });
    latticeRows.forEach((row) => {
      const rect = add(node, "rect", {
        x: sx(row.s_start_m),
        width: Math.max(1, sx(row.s_end_m) - sx(row.s_start_m)),
        y: Math.min(syLattice(row.y0), syLattice(row.y1)),
        height: Math.max(2, Math.abs(syLattice(row.y1) - syLattice(row.y0))),
        fill: row.color,
        opacity: row.family === "Drift" ? .35 : .92,
      });
      add(rect, "title", {}, `${row.name} · ${row.family} · ${formatValue(row.length_m, 5)} m`);
    });
    add(node, "text", {
      x: (margins.left + plotRight) / 2, y: 433, "text-anchor": "middle",
    }, "Longitudinal position s (m)");

    const legend = [
      { label: "βx", color: "#b3472e" },
      { label: "βy", color: "#2f6f9f" },
      { label: "Dx", color: "#2f7655" },
      ...[...new Map(latticeRows.map((row) => [row.family, row.color])).entries()]
        .map(([label, color]) => ({ label, color })),
    ];
    return chartContainer(node, legend, data.meta);
  }

  function radiationView(data) {
    const container = document.createElement("div");
    const metrics = document.createElement("div"); metrics.className = "metrics-grid";
    metrics.innerHTML = Object.entries(data.meta || {}).map(([key, value]) => `<div class="metric"><span class="label">${escapeHtml(key.replaceAll("_", " "))}</span><span class="value">${escapeHtml(formatValue(value))}</span></div>`).join("");
    container.appendChild(metrics);
    const rows = (data.integrals || []).map((row) => ({ Quantity: row.integral, Value: formatValue(row.value), Unit: row.unit }));
    container.appendChild(makeTable(rows));
    if (data.partitions?.length) {
      const node = svg(820, 190), max = Math.max(1, ...data.partitions.map((r) => Number(r.partition)));
      data.partitions.forEach((row, index) => {
        const y = 24 + index * 48;
        add(node, "text", { x: 5, y: y + 14 }, row.plane);
        add(node, "rect", { x: 105, y, width: 650 * Number(row.partition) / max, height: 22, fill: ["#b3472e", "#2f6f9f", "#2f7655"][index] });
        add(node, "text", { x: 112 + 650 * Number(row.partition) / max, y: y + 15 }, formatValue(row.partition, 4));
      });
      container.appendChild(node);
    }
    return container;
  }

  function chromaticityChart(data) {
    const rows = data.rows || [], node = svg(820, 270), colors = { Natural: "#999", Target: "#d29a4a", Corrected: "#2f7655" };
    const transform = (v) => Math.sign(Number(v)) * Math.log10(1 + Math.abs(Number(v)));
    const values = rows.map((r) => transform(r.chromaticity)), yd = domain([...values, 0], .12), sy = scale(yd, [220, 20]);
    add(node, "line", { x1: 45, x2: 800, y1: sy(0), y2: sy(0), class: "axis" });
    rows.forEach((row, index) => {
      const x = 60 + index * (720 / Math.max(rows.length, 1)), value = transform(row.chromaticity), y = Math.min(sy(0), sy(value));
      add(node, "rect", { x, y, width: Math.max(24, 570 / Math.max(rows.length, 1)), height: Math.max(1, Math.abs(sy(value) - sy(0))), fill: colors[row.state] || "#777" });
      add(node, "text", { x: x + 18, y: 247, "text-anchor": "middle" }, `${row.plane} ${row.state[0]}`);
    });
    add(node, "text", { x: 15, y: 135, transform: "rotate(-90 15 135)", "text-anchor": "middle" }, "signed log₁₀(1 + |ξ|)");
    return chartContainer(node, Object.entries(colors).map(([label,color]) => ({label,color})), data.meta);
  }

  function apertureChart(data) {
    const rows = data.rows || [], node = svg(820, 380);
    const xd = domain(rows.map((r) => r.x_mm), .08), yd = domain(rows.map((r) => r.y_mm), .08);
    const { sx, sy } = axes(node, xd, yd, 820, 380, { x: "x (mm)", y: "y (mm)" });
    const colors = ["#b3472e", "#2f6f9f", "#2f7655", "#9c6a1c", "#73558f"];
    const scans = [...new Set(rows.map((r) => r.scan))];
    scans.forEach((scan, index) => {
      const points = rows.filter((r) => r.scan === scan).sort((a,b) => a.point_order - b.point_order);
      add(node, "path", { d: linePath(points, "x_mm", "y_mm", sx, sy), fill: "none", stroke: colors[index % colors.length], "stroke-width": 2 });
    });
    return chartContainer(node, scans.map((scan, index) => ({ label: rows.find((r) => r.scan === scan)?.momentum_offset || scan, color: colors[index % colors.length] })), data.meta);
  }

  function robustnessChart(data) {
    const rows = data.rows || [], node = svg(820, 320), states = [...new Set(rows.map((r) => r.state))];
    const transformed = rows.map((r) => ({ ...r, log: Math.log10(Math.max(Number(r.rms_um), 1e-6)) }));
    const yd = domain([...transformed.map((r) => r.log), 0], .1), sy = scale(yd, [270, 20]);
    const seeds = [...new Set(rows.map((r) => r.seed))], group = 720 / Math.max(seeds.length, 1), bar = Math.min(18, group / 3);
    const colors = { "Before correction": "#b3472e", "After correction": "#2f7655" };
    transformed.forEach((row) => {
      const i = seeds.indexOf(row.seed), j = states.indexOf(row.state), x = 65 + i * group + j * bar;
      add(node, "rect", { x, y: sy(row.log), width: bar - 2, height: 270 - sy(row.log), fill: colors[row.state] || "#777" });
    });
    seeds.forEach((seed, i) => add(node, "text", { x: 65 + i * group + bar, y: 293, "text-anchor": "middle" }, seed));
    add(node, "text", { x: 410, y: 314, "text-anchor": "middle" }, "Error seed");
    add(node, "text", { x: 13, y: 150, transform: "rotate(-90 13 150)", "text-anchor": "middle" }, "log₁₀ orbit RMS (µm)");
    return chartContainer(node, Object.entries(colors).map(([label,color]) => ({label,color})), data.meta);
  }

  function comparisonTable(data) {
    const container = document.createElement("div");
    container.appendChild(makeTable(data.rows || []));
    const meta = document.createElement("div"); meta.className = "chart-meta";
    meta.innerHTML = Object.entries(data.meta || {}).map(([key,value]) => `<span>${escapeHtml(key.replaceAll("_", " "))}: <b>${escapeHtml(formatValue(value))}</b></span>`).join("");
    container.appendChild(meta);
    return container;
  }

  function datasetCatalogView(data) {
    const rows = (data.rows || []).map((row) => ({
      Dataset: row.id,
      Title: row.title,
      Rows: row.row_count,
      Scope: row.scope?.kind || "archived rows",
      Columns: (row.columns || []).map((column) =>
        `${column.name}${column.unit ? ` [${column.unit}]` : ""}`
      ).join(", "),
      Accepted: row.accepted,
      Source: row.source,
    }));
    return comparisonTable({ rows, meta: data.meta });
  }

  function dataView(data) {
    const container = document.createElement("div");
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const view = data.view || { kind: "table" };
    if (view.kind === "table") {
      container.appendChild(makeTable(rows));
    } else {
      const xKey = view.x;
      const yKeys = Array.isArray(view.y) ? view.y : [view.y].filter(Boolean);
      const finiteRows = rows.filter((row) =>
        Number.isFinite(Number(row[xKey])) && yKeys.some((key) => Number.isFinite(Number(row[key])))
      );
      const node = svg(820, 330);
      const xDomain = domain(finiteRows.map((row) => row[xKey]), .05);
      const yDomain = domain(finiteRows.flatMap((row) => yKeys.map((key) => row[key])), .08);
      const { sx, sy } = axes(node, xDomain, yDomain, 820, 330, {
        x: view.x_label || xKey,
        y: view.y_label || yKeys.join(", "),
      });
      const colors = ["#b3472e", "#2f6f9f", "#2f7655", "#9c6a1c", "#73558f", "#777"];
      const groups = [];
      yKeys.forEach((yKey) => {
        const seriesValues = view.series
          ? [...new Set(finiteRows.map((row) => String(row[view.series])))]
          : [""];
        seriesValues.forEach((seriesValue) => groups.push({
          yKey,
          seriesValue,
          label: seriesValue ? `${yKey} · ${seriesValue}` : yKey,
          rows: finiteRows.filter((row) =>
            (!view.series || String(row[view.series]) === seriesValue)
            && Number.isFinite(Number(row[yKey]))
          ).sort((left, right) => Number(left[xKey]) - Number(right[xKey])),
        }));
      });
      groups.forEach((group, index) => {
        const color = colors[index % colors.length];
        if (view.kind === "scatter") {
          group.rows.forEach((row) => add(node, "circle", {
            cx: sx(row[xKey]), cy: sy(row[group.yKey]), r: 2.8, fill: color, opacity: .82,
          }));
        } else if (view.kind === "bar") {
          const baseline = sy(Math.max(0, yDomain[0]));
          const width = Math.max(2, 620 / Math.max(group.rows.length * groups.length, 1));
          group.rows.forEach((row) => add(node, "rect", {
            x: sx(row[xKey]) - width / 2,
            y: Math.min(baseline, sy(row[group.yKey])),
            width,
            height: Math.max(1, Math.abs(sy(row[group.yKey]) - baseline)),
            fill: color,
          }));
        } else {
          add(node, "path", {
            d: linePath(group.rows, xKey, group.yKey, sx, sy),
            fill: "none", stroke: color, "stroke-width": 1.8,
          });
        }
      });
      container.appendChild(chartContainer(
        node,
        groups.map((group, index) => ({ label: group.label, color: colors[index % colors.length] })),
        data.meta,
      ));
    }
    if (view.kind === "table") {
      const meta = document.createElement("div"); meta.className = "chart-meta";
      meta.innerHTML = Object.entries(data.meta || {}).map(([key, value]) =>
        `<span>${escapeHtml(key.replaceAll("_", " "))}: <b>${escapeHtml(formatValue(value))}</b></span>`
      ).join("");
      container.appendChild(meta);
    }
    return container;
  }

  function researchPlots(data) {
    const container = document.createElement("div"), colors = ["#b3472e", "#2f6f9f", "#2f7655", "#9c6a1c", "#73558f"];
    (data.groups || []).forEach((group) => {
      const title = document.createElement("div"); title.className = "card-title"; title.textContent = group.title; container.appendChild(title);
      const rows = group.rows || [], node = svg(820, 300), xd = domain(rows.map((r) => r.x)), yd = domain(rows.map((r) => r.y));
      const { sx, sy } = axes(node, xd, yd, 820, 300, { x: group.x_label, y: group.y_label });
      const series = [...new Set(rows.map((r) => r.series))];
      series.forEach((name, index) => {
        const points = rows.filter((r) => r.series === name).sort((a,b) => a.x - b.x);
        add(node, "path", { d: linePath(points, "x", "y", sx, sy), fill: "none", stroke: colors[index % colors.length], "stroke-width": 1.8 });
      });
      container.appendChild(chartContainer(node, series.map((label,index) => ({ label, color: colors[index % colors.length] }))));
    });
    return container;
  }

  function particleTrackingCharts(data) {
    const container = document.createElement("div");
    const rows = (data.rows || []).filter((row) => Number.isFinite(Number(row.turn)) && !row.lost);
    const particles = [...new Set(rows.map((row) => row.particle))];
    const colors = ["#b3472e", "#2f6f9f", "#2f7655", "#9c6a1c", "#73558f", "#777"];
    const definitions = {
      x_turn: { x: "turn", y: "x_mm", xLabel: "Turn", yLabel: "x (mm)", title: "Horizontal turn-by-turn trajectory" },
      y_turn: { x: "turn", y: "y_mm", xLabel: "Turn", yLabel: "y (mm)", title: "Vertical turn-by-turn trajectory" },
      x_px: { x: "x_mm", y: "px_mrad", xLabel: "x (mm)", yLabel: "px (mrad)", title: "Horizontal phase space [x, px]" },
      y_py: { x: "y_mm", y: "py_mrad", xLabel: "y (mm)", yLabel: "py (mrad)", title: "Vertical phase space [y, py]" },
    };
    const requestedViews = data.meta?.views?.length ? data.meta.views : ["x_turn", "y_turn"];
    for (const view of requestedViews) {
      const definition = definitions[view];
      if (!definition) continue;
      const viewRows = rows.filter((row) =>
        Number.isFinite(Number(row[definition.x])) && Number.isFinite(Number(row[definition.y]))
      );
      const title = document.createElement("div");
      title.className = "card-title"; title.textContent = definition.title;
      container.appendChild(title);
      const node = svg(820, 275);
      const xd = domain(viewRows.map((row) => row[definition.x]), definition.x === "turn" ? 0 : .08);
      const yd = domain(viewRows.map((row) => row[definition.y]), .08);
      const { sx, sy } = axes(node, xd, yd, 820, 275, { x: definition.xLabel, y: definition.yLabel });
      particles.forEach((particle, index) => {
        const points = viewRows.filter((row) => row.particle === particle).sort((a,b) => a.turn - b.turn);
        add(node, "path", {
          d: linePath(points, definition.x, definition.y, sx, sy), fill: "none",
          stroke: colors[index % colors.length], "stroke-width": 1.5,
        });
      });
      container.appendChild(chartContainer(
        node,
        particles.map((label,index) => ({ label, color: colors[index % colors.length] })),
        view === requestedViews[0] ? data.meta : {},
      ));
    }
    if (data.survival?.length) {
      const title = document.createElement("div");
      title.className = "card-title"; title.textContent = "Particle survival";
      container.appendChild(title);
      container.appendChild(makeTable(data.survival));
    }
    return container;
  }

  function populationOptimizationView(data) {
    const container = document.createElement("div");
    const algorithm = data.algorithm || {};
    const method = document.createElement("p"); method.className = "artifact-method";
    method.textContent = [
      algorithm.name,
      algorithm.proposal,
      algorithm.environmental_selection,
    ].filter(Boolean).join(" · ");
    if (method.textContent) container.appendChild(method);
    const metrics = document.createElement("div"); metrics.className = "metrics-grid";
    metrics.innerHTML = Object.entries(data.meta || {}).filter(([, value]) => value !== null && value !== undefined)
      .map(([key, value]) => `<div class="metric"><span class="label">${escapeHtml(key.replaceAll("_", " "))}</span><span class="value">${escapeHtml(formatValue(value))}</span></div>`).join("");
    container.appendChild(metrics);
    const groups = (data.plots || []).map((plot) => ({
      title: plot.title,
      x_label: plot.x_label,
      y_label: plot.y_label,
      rows: plot.rows || [],
      kind: plot.kind || "line",
    }));
    const colors = ["#b3472e", "#2f6f9f", "#2f7655", "#9c6a1c", "#73558f", "#777"];
    groups.forEach((group) => {
      const rows = (group.rows || []).filter((row) => Number.isFinite(Number(row.x)) && Number.isFinite(Number(row.y)));
      if (!rows.length) return;
      const title = document.createElement("div"); title.className = "card-title";
      title.textContent = group.title || "Optimization result"; container.appendChild(title);
      const node = svg(820, 300), xd = domain(rows.map((row) => row.x)), yd = domain(rows.map((row) => row.y));
      const { sx, sy } = axes(node, xd, yd, 820, 300, { x: group.x_label, y: group.y_label });
      const series = [...new Set(rows.map((row) => row.series || "result"))];
      series.forEach((name, index) => {
        const points = rows.filter((row) => (row.series || "result") === name).sort((a,b) => Number(a.x) - Number(b.x));
        if (group.kind === "scatter") {
          points.forEach((point) => add(node, "circle", { cx: sx(point.x), cy: sy(point.y), r: 3, fill: colors[index % colors.length] }));
        } else {
          add(node, "path", { d: linePath(points, "x", "y", sx, sy), fill: "none", stroke: colors[index % colors.length], "stroke-width": 1.7 });
        }
      });
      container.appendChild(chartContainer(node, series.map((label, index) => ({ label, color: colors[index % colors.length] }))));
    });
    const selectedOptics = Array.isArray(data.optics)
      ? { rows: data.optics, meta: { scope: data.optics_scope } }
      : data.optics;
    if (selectedOptics?.rows?.length) {
      const title = document.createElement("div"); title.className = "card-title";
      title.textContent = `Selected optics — ${data.optics_scope || "one periodic cell"}`;
      container.appendChild(title);
      container.appendChild(opticsChart(selectedOptics));
    }
    (data.tables || []).forEach((table) => {
      const title = document.createElement("div"); title.className = "card-title";
      title.textContent = table.title || "Optimization table"; container.appendChild(title);
      container.appendChild(makeTable(table.rows || []));
    });
    return container;
  }

  function codedExperimentView(data) {
    const container = document.createElement("div");
    if (data.method) {
      const method = document.createElement("p"); method.className = "artifact-method";
      method.textContent = data.method; container.appendChild(method);
    }
    const summary = data.summary && typeof data.summary === "object"
      ? Object.entries(data.summary).map(([quantity, value]) => ({ quantity, value })) : [];
    if (summary.length) {
      const title = document.createElement("div"); title.className = "card-title";
      title.textContent = "Computed summary"; container.appendChild(title);
      container.appendChild(makeTable(summary));
    }
    (data.tables || []).forEach((table) => {
      const title = document.createElement("div"); title.className = "card-title";
      title.textContent = table.title || "Computed table"; container.appendChild(title);
      container.appendChild(makeTable(table.rows || []));
    });
    const colors = ["#b3472e", "#2f6f9f", "#2f7655", "#9c6a1c", "#73558f", "#777"];
    (data.plots || []).forEach((plot) => {
      const rows = (plot.rows || []).filter((row) =>
        Number.isFinite(Number(row.x)) && Number.isFinite(Number(row.y))
      );
      const title = document.createElement("div"); title.className = "card-title";
      title.textContent = plot.title || "Computed plot"; container.appendChild(title);
      const node = svg(820, 300), xd = domain(rows.map((row) => row.x)), yd = domain(rows.map((row) => row.y));
      const { sx, sy } = axes(node, xd, yd, 820, 300, {
        x: plot.x_label || "x", y: plot.y_label || "y",
      });
      const series = [...new Set(rows.map((row) => row.series || "result"))];
      series.forEach((name, index) => {
        const points = rows.filter((row) => (row.series || "result") === name);
        if ((plot.kind || "line") === "scatter") {
          points.forEach((point) => add(node, "circle", {
            cx: sx(point.x), cy: sy(point.y), r: 2.4, fill: colors[index % colors.length],
          }));
        } else {
          add(node, "path", {
            d: linePath(points, "x", "y", sx, sy), fill: "none",
            stroke: colors[index % colors.length], "stroke-width": 1.6,
          });
        }
      });
      container.appendChild(chartContainer(
        node, series.map((label, index) => ({ label, color: colors[index % colors.length] })),
        data.meta || {},
      ));
    });
    const notes = [];
    const conventions = Array.isArray(data.conventions)
      ? data.conventions
      : Object.entries(data.conventions || {}).map(([key, value]) => `${key.replaceAll("_", " ")}: ${formatValue(value)}`);
    conventions.forEach((item) => notes.push(`Convention: ${item}`));
    const limits = Array.isArray(data.model_limits) ? data.model_limits : [data.model_limits].filter(Boolean);
    limits.forEach((item) => notes.push(`Limit: ${item}`));
    if (notes.length) {
      const list = document.createElement("ul"); list.className = "coded-notes";
      notes.forEach((item) => { const row = document.createElement("li"); row.textContent = item; list.appendChild(row); });
      container.appendChild(list);
    }
    if (data.claim_boundary) {
      const boundary = document.createElement("div"); boundary.className = "code-claim-boundary";
      boundary.textContent = data.claim_boundary; container.appendChild(boundary);
    }
    if (data.code) {
      const details = document.createElement("details"); details.className = "generated-code";
      const summaryNode = document.createElement("summary"); summaryNode.textContent = "Generated JuTrack code";
      const pre = document.createElement("pre"); pre.textContent = data.code;
      details.append(summaryNode, pre); container.appendChild(details);
    }
    return container;
  }

  function makeTable(rows) {
    const wrap = document.createElement("div"); wrap.className = "chart-wrap";
    if (!rows.length) { wrap.textContent = "No tabular values were recorded."; return wrap; }
    const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))].slice(0, 12);
    const table = document.createElement("table"); table.className = "data-table";
    table.innerHTML = `<thead><tr>${columns.map((column) => `<th>${escapeHtml(column.replaceAll("_", " "))}</th>`).join("")}</tr></thead>
      <tbody>${rows.slice(0, 100).map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(formatValue(row[column]))}</td>`).join("")}</tr>`).join("")}</tbody>`;
    wrap.appendChild(table); return wrap;
  }

  function nearBottom() { return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 260; }
  function scrollBottom(force = false) {
    if (force || nearBottom()) window.scrollTo({ top: document.documentElement.scrollHeight, behavior: force ? "auto" : "smooth" });
  }

  function autoResize() {
    els.prompt.style.height = "auto";
    els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 180)}px`;
    els.send.disabled = state.busy || !els.prompt.value.trim();
  }

  async function submitMessage(event) {
    event.preventDefault();
    const content = els.prompt.value.trim();
    if (!content || state.busy) return;
    ensureNotificationPermission();
    state.busy = true; autoResize();
    els.prompt.value = ""; autoResize();
    try {
      if (!state.projectId) {
        els.welcome.classList.add("hidden");
        els.conversation.classList.remove("hidden");
        els.conversation.replaceChildren(renderMessage({ id: "optimistic", role: "user", content, timestamp: new Date().toISOString(), attachments: [] }), renderLive({ label: "Creating the research contract and starting JuTrack" }));
        scrollBottom(true);
        const result = await api("/api/projects", { method: "POST", body: JSON.stringify({ goal: content, context: els.context.value.trim() }) });
        await loadProjects();
        await openProject(result.project.id);
      } else {
        const optimistic = renderMessage({ id: "optimistic", role: "user", content, timestamp: new Date().toISOString(), attachments: [] });
        els.conversation.appendChild(optimistic); scrollBottom();
        await api(`/api/projects/${encodeURIComponent(state.projectId)}/messages`, { method: "POST", body: JSON.stringify({ content }) });
        applySnapshot(await api(`/api/projects/${encodeURIComponent(state.projectId)}`));
      }
    } catch (error) {
      els.prompt.value = content;
      showToast(error.message);
      if (!state.projectId) resetToNew();
    } finally {
      state.busy = false; autoResize(); els.prompt.focus();
    }
  }

  function toggleWorkspace(open) {
    $("#workspace-panel").classList.toggle("hidden", !open);
    document.body.classList.toggle("workspace-open", open);
    $("#workspace-toggle").setAttribute("aria-expanded", String(open));
  }

  function resetWorkspace() {
    state.programName = null;
    state.messageNodes.clear();
    $("#program-editor").classList.add("hidden");
    $("#program-source").value = "";
    $("#pause-agent").classList.add("hidden");
    $("#program-list").replaceChildren();
    $("#program-list").dataset.signature = "";
    $("#activity-log").replaceChildren();
    $("#activity-log").dataset.signature = "";
  }

  function updateWorkspace(snapshot) {
    const session = snapshot.session || {};
    const pause = $("#pause-agent");
    pause.classList.toggle("hidden", !snapshot.live || !session.can_pause);
    pause.disabled = Boolean(session.pause_requested);
    pause.textContent = session.pause_requested ? "Pausing…" : "Pause agent";
    $("#composer-hint").textContent = snapshot.live
      ? `${snapshot.live.pending_count || 0} queued · Send to steer the active session`
      : "Enter to send · Shift+Enter for newline";
    const list = $("#program-list");
    const files = session.programs || [];
    const signature = JSON.stringify(files);
    if (list.dataset.signature !== signature) {
      list.dataset.signature = signature;
      list.replaceChildren(...files.map((file) => {
        const button = document.createElement("button");
        button.textContent = file.name;
        button.addEventListener("click", () => openProgram(file.name));
        return button;
      }));
      if (!files.length) list.textContent = "No saved research programs yet.";
    }
    const log = $("#activity-log");
    const activitySignature = JSON.stringify(session.activities || []);
    if (log.dataset.signature !== activitySignature) {
      log.dataset.signature = activitySignature;
      log.innerHTML = (session.activities || []).map((item) => `<details ${item.status === "failed" ? "open" : ""}><summary>${activityTitle(item)} — ${escapeHtml(item.status)}</summary><pre>${escapeHtml(item.detail || "Completed without a reported error.")}</pre></details>`).join("") || "No tool activity yet.";
    }
  }

  async function openProgram(name) {
    const projectId = state.projectId;
    if (!projectId) return;
    try {
      const result = await api(`/api/projects/${encodeURIComponent(projectId)}/programs/${encodeURIComponent(name)}`);
      if (projectId !== state.projectId) return;
      state.programName = name;
      $("#program-editor").classList.remove("hidden");
      $("#program-name").value = name.replace(/\.py$/, "_draft.py");
      $("#program-source").value = result.source;
      $("#program-status").textContent = `Viewing ${name}. Saving creates a new file; nothing runs automatically.`;
    } catch (error) { showToast(error.message); }
  }

  $("#workspace-toggle").addEventListener("click", () => toggleWorkspace($("#workspace-panel").classList.contains("hidden")));
  $("#workspace-close").addEventListener("click", () => toggleWorkspace(false));
  $("#show-programs").addEventListener("click", () => { $("#program-list").classList.remove("hidden"); $("#activity-log").classList.add("hidden"); });
  $("#show-activity").addEventListener("click", () => { $("#program-list").classList.add("hidden"); $("#activity-log").classList.remove("hidden"); });
  $("#pause-agent").addEventListener("click", async () => {
    try {
      const result = await api(`/api/projects/${encodeURIComponent(state.projectId)}/pause`, { method: "POST", body: "{}" });
      showToast(result.detail);
    } catch (error) { showToast(error.message); }
  });
  $("#save-program").addEventListener("click", async () => {
    const projectId = state.projectId;
    if (!projectId) return;
    try {
      const result = await api(`/api/projects/${encodeURIComponent(projectId)}/programs`, { method: "POST", body: JSON.stringify({ name: $("#program-name").value, source: $("#program-source").value }) });
      if (projectId !== state.projectId) return;
      $("#program-status").textContent = `Saved programs/${result.name}. Ask the agent to review or run this draft.`;
      applySnapshot(await api(`/api/projects/${encodeURIComponent(projectId)}`));
    } catch (error) { showToast(error.message); }
  });

  els.composer.addEventListener("submit", submitMessage);
  setInterval(() => {
    const counter = $("[data-live-elapsed]");
    if (counter && state.snapshot?.live) {
      const elapsed = Number(state.snapshot.live.progress?.elapsed_seconds || 0) + (Date.now() - state.snapshotReceivedAt) / 1000;
      counter.textContent = `${elapsedLabel(elapsed)} elapsed`;
    }
  }, 1000);
  els.prompt.addEventListener("input", autoResize);
  els.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault(); els.composer.requestSubmit();
    }
  });
  $("#new-chat").addEventListener("click", resetToNew);
  $("#open-sidebar").addEventListener("click", () => els.sidebar.classList.add("open"));
  $("#close-sidebar").addEventListener("click", () => els.sidebar.classList.remove("open"));
  $("#context-toggle").addEventListener("click", () => {
    els.context.classList.toggle("hidden");
    if (!els.context.classList.contains("hidden")) els.context.focus();
  });
  document.querySelectorAll(".example").forEach((button) => button.addEventListener("click", () => {
    els.prompt.value = button.dataset.prompt; autoResize(); els.prompt.focus();
  }));
  window.addEventListener("popstate", () => {
    const match = location.pathname.match(/^\/p\/([^/]+)$/);
    match ? openProject(decodeURIComponent(match[1]), false) : resetToNew();
  });

  async function boot() {
    await loadProjects();
    const match = location.pathname.match(/^\/p\/([^/]+)$/);
    if (match) await openProject(decodeURIComponent(match[1]), false);
    else resetToNew();
  }
  boot();
})();
