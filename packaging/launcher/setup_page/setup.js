"use strict";

/* Setup-page state machine. Python pushes events via qw35OnEvent(); the page
 * calls back through window.pywebview.api.*. */

let state = null;          // last get_state() payload
let probes = {};           // name -> {size, available} (HEAD results)
let retryAction = null;

const $ = (id) => document.getElementById(id);
const sections = ["checking", "consent", "downloading", "starting", "error"];

function show(id) {
  for (const s of sections) $(s).hidden = s !== id;
}

function humanBytes(n) {
  if (n == null) return "?";
  const units = ["B", "KB", "MB", "GB"];
  let u = 0;
  while (n >= 1024 && u < units.length - 1) { n /= 1024; u++; }
  return n.toFixed(u >= 2 ? 1 : 0) + " " + units[u];
}

function humanTime(s) {
  if (s == null || !isFinite(s)) return "—";
  if (s < 60) return Math.round(s) + " s";
  if (s < 3600) return Math.round(s / 60) + " min";
  return (s / 3600).toFixed(1) + " h";
}

/* ---- consent ------------------------------------------------------------- */

function modelSize(m) {
  const p = probes[m.name];
  return (p && p.size) || m.fallback_size;
}

function modelAvailable(m) {
  const p = probes[m.name];
  return !p || p.available;
}

function missingModels() {
  return state.models.filter((m) => !m.present && modelAvailable(m));
}

function renderConsent() {
  show("consent");
  const table = $("model-table");
  table.innerHTML = "";
  let totalToGet = 0;
  let resumed = 0;
  for (const m of state.models) {
    const row = table.insertRow();
    row.className = m.present ? "present" : "";
    row.insertCell().textContent = m.label;
    if (m.present) {
      row.insertCell().textContent = "already on this Mac ✓";
    } else if (!modelAvailable(m)) {
      row.insertCell().textContent = "unavailable — skipped*";
    } else {
      row.insertCell().textContent = humanBytes(modelSize(m));
      totalToGet += modelSize(m) - m.part_size;
      resumed += m.part_size;
    }
  }
  const totalRow = table.insertRow();
  totalRow.className = "total";
  totalRow.insertCell().textContent = "To download";
  totalRow.insertCell().textContent = humanBytes(totalToGet);

  $("disk-line").textContent =
    "Free disk space: " + humanBytes(state.disk_free) +
    " (the download needs " + humanBytes(totalToGet) + " plus a 2 GB margin)";

  const mbits = (speed) => humanTime((totalToGet * 8) / (speed * 1e6));
  $("eta-line").textContent =
    "Estimated download time: ≈" + mbits(50) + " at 50 Mbit/s, ≈" +
    mbits(200) + " at 200 Mbit/s, ≈" + mbits(500) + " at 500 Mbit/s";

  const resumeLine = $("resume-line");
  resumeLine.hidden = resumed === 0;
  if (resumed > 0) {
    resumeLine.textContent =
      humanBytes(resumed) + " from an earlier attempt is already on disk — " +
      "the download resumes from there.";
  }

  const skipped = state.models.filter((m) => !m.present && !modelAvailable(m));
  $("skip-line").hidden = skipped.length === 0;
  if (skipped.length > 0) {
    $("skip-line").textContent =
      "* " + skipped.map((m) => m.label).join(", ") + " cannot be downloaded " +
      "without a Hugging Face login right now. qw35 works without it — " +
      "web-result ranking just falls back to a simpler method.";
  }
  $("download-btn").textContent =
    (resumed > 0 ? "Resume download" : "Download") + " (" + humanBytes(totalToGet) + ")";
  $("gguf-dir").textContent = state.gguf_dir;
}

/* ---- downloading --------------------------------------------------------- */

function renderDownloadBlocks() {
  const box = $("file-progress");
  box.innerHTML = "";
  for (const m of missingModels()) {
    const size = modelSize(m);
    const block = document.createElement("div");
    block.className = "file-block";
    block.id = "file-" + m.name;
    block.innerHTML =
      '<div class="file-head"><span class="file-name"></span>' +
      '<span class="file-stats">waiting…</span></div>' +
      '<div class="bar"><div></div></div>';
    block.querySelector(".file-name").textContent = m.label;
    const pct = size ? (m.part_size / size) * 100 : 0;
    block.querySelector(".bar > div").style.width = pct + "%";
    box.appendChild(block);
  }
}

function onProgress(ev) {
  const block = $("file-" + ev.file);
  if (!block) return;
  const pct = ev.total ? (ev.received / ev.total) * 100 : 0;
  block.querySelector(".bar > div").style.width = pct.toFixed(2) + "%";
  block.querySelector(".file-stats").textContent =
    humanBytes(ev.received) + " / " + humanBytes(ev.total) +
    " · " + humanBytes(ev.speed_bps) + "/s · " + humanTime(ev.eta_s) + " left";
}

function onFileDone(ev) {
  const block = $("file-" + ev.file);
  if (!block) return;
  block.classList.add("done");
  block.querySelector(".bar > div").style.width = "100%";
  block.querySelector(".file-stats").textContent = "done ✓";
}

/* ---- starting ------------------------------------------------------------ */

function renderStarting(phase) {
  show("starting");
  if (phase === "engine-starting") {
    $("starting-title").textContent = "Starting the qw35 engine";
    $("starting-detail").textContent =
      "Loading Qwowl3.5-9B (~5 GB) into memory — this can take a minute…";
  } else {
    $("starting-title").textContent = "Starting the agent interface";
    $("starting-detail").textContent = "Almost there…";
  }
}

/* ---- error --------------------------------------------------------------- */

function renderError(message, retriable, action) {
  show("error");
  $("error-message").textContent = message;
  $("retry-btn").hidden = !retriable;
  retryAction = retriable ? action : null;
}

/* ---- event pump (called from Python) ------------------------------------- */

function onFileSkipped(ev) {
  const block = $("file-" + ev.file);
  if (!block) return;
  block.classList.add("done");
  block.querySelector(".file-stats").textContent =
    "skipped (needs a Hugging Face login) — qw35 works without it";
}

window.qw35OnEvent = function (ev) {
  switch (ev.type) {
    case "progress": onProgress(ev); break;
    case "file-done": onFileDone(ev); break;
    case "file-skipped": onFileSkipped(ev); break;
    case "phase":
      switch (ev.value) {
        case "downloading": show("downloading"); break;
        case "downloaded": launch(); break;
        case "cancelled": refreshState().then(renderConsent); break;
        case "engine-starting":
        case "agent-starting": renderStarting(ev.value); break;
        case "ready": renderStarting("agent-starting"); break; // load_url follows
      }
      break;
    case "error":
      renderError(ev.message, ev.retriable, lastAction);
      break;
  }
};

/* The action to retry after an error: download errors retry the download,
 * launch errors are terminal (retriable=false), so tracking the last started
 * action is enough. */
let lastAction = startDownloads;

function startDownloads() {
  lastAction = startDownloads;
  renderDownloadBlocks();
  show("downloading");
  window.pywebview.api.start_downloads();
}

function launch() {
  lastAction = launch;
  renderStarting("engine-starting");
  window.pywebview.api.launch();
}

async function refreshState() {
  state = await window.pywebview.api.get_state();
  return state;
}

/* ---- boot ---------------------------------------------------------------- */

async function boot() {
  show("checking");
  await refreshState();
  if (missingModels().length === 0) {
    launch();
    return;
  }
  renderConsent();
  // Refine sizes/availability with real HEAD results in the background.
  window.pywebview.api.probe_models().then((real) => {
    probes = real || {};
    if ($("consent").hidden) return;
    if (missingModels().length === 0) {
      launch();  // everything still missing turned out to be skippable
      return;
    }
    renderConsent();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  $("download-btn").addEventListener("click", startDownloads);
  $("cancel-btn").addEventListener("click", () => window.pywebview.api.cancel_downloads());
  $("quit-btn").addEventListener("click", () => window.pywebview.api.quit());
  $("error-quit-btn").addEventListener("click", () => window.pywebview.api.quit());
  $("retry-btn").addEventListener("click", () => { if (retryAction) retryAction(); });
  $("reveal-link").addEventListener("click", (e) => {
    e.preventDefault();
    window.pywebview.api.reveal_models();
  });
});

window.addEventListener("pywebviewready", boot);
