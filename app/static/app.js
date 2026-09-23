/* AI Video Attendance - basic web UI (vanilla JS) */
"use strict";

/* ------------------------------------------------------------- helpers */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtDT(s) {
  if (!s) return "—";
  try {
    const d = new Date(String(s).replace(" ", "T"));
    return isNaN(d) ? String(s) : d.toLocaleString();
  } catch { return String(s); }
}

function fmtTime(s) {
  if (!s) return "—";
  try {
    const d = new Date(String(s).replace(" ", "T"));
    return isNaN(d) ? String(s) : d.toLocaleTimeString();
  } catch { return String(s); }
}

function fmtDur(sec) {
  if (sec == null) return "—";
  sec = Number(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

function localToday() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function toast(msg, kind = "info") {
  const el = document.createElement("div");
  el.className = "toast " + (kind === "error" ? "err" : kind === "ok" ? "ok" : "");
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 3500);
}

/* ------------------------------------------------------------- auth/api */
const TOKEN_KEY = "iva_token";

const getToken = () => localStorage.getItem(TOKEN_KEY) || "";
const setToken = (t) => localStorage.setItem(TOKEN_KEY, t);

function logout() {
  localStorage.removeItem(TOKEN_KEY);
  try { if (ws) ws.close(); } catch {}
  $("#app-view").hidden = true;
  $("#login-view").hidden = false;
}

async function api(path, opts = {}) {
  const headers = {};
  const tok = getToken();
  if (tok) headers["Authorization"] = "Bearer " + tok;

  let body = opts.body;
  if (opts.form) {
    body = opts.form; // FormData -> browser sets multipart boundary
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }

  const res = await fetch(path, {
    method: opts.method || (body ? "POST" : "GET"),
    headers, body,
  });

  if (res.status === 401) {
    logout();
    throw new Error("Session expired — please log in again");
  }
  const txt = await res.text();
  let data = null;
  try { data = txt ? JSON.parse(txt) : null; } catch { data = { detail: txt }; }
  if (!res.ok) {
    let d = data && data.detail;
    if (Array.isArray(d)) d = d.map((e) => (e && e.msg) || JSON.stringify(e)).join("; ");
    const err = new Error(d || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

function errToast(e) { toast(e && e.message ? e.message : String(e), "error"); }

/* ------------------------------------------------------------- login */
async function tryLogin(ev) {
  ev.preventDefault();
  const errEl = $("#login-error");
  errEl.hidden = true;
  try {
    const data = await api("/auth/token", {
      body: {
        username: $("#login-username").value.trim(),
        password: $("#login-password").value,
      },
    });
    setToken(data.access_token);
    enterApp();
  } catch (e) {
    errEl.textContent = e.message;
    errEl.hidden = false;
  }
}

async function boot() {
  if (getToken()) { enterApp(); return; }
  // auth may be disabled (AUTH_REQUIRED=false) -> probe an authed endpoint
  try {
    await api("/students?limit=1");
    enterApp();
    return;
  } catch (e) {
    if (e && e.status === 401) { /* needs login */ }
    else if (!(e && e.message && e.message.includes("log in"))) { enterApp(); }
  }
  $("#login-view").hidden = false;
}

function enterApp() {
  $("#login-view").hidden = true;
  $("#app-view").hidden = false;
  $("#user-label").textContent = getToken() ? "admin" : "auth disabled";
  loadDashboard();
  connectWS();
}

/* ------------------------------------------------------------- sections */
const SECTION_TITLES = {
  dashboard: "Dashboard", students: "Students", cameras: "Cameras",
  attendance: "Attendance", live: "Live", system: "System",
};
let liveTimer = null;

function showSection(name) {
  $$(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.section === name));
  $$("main section").forEach((s) => { s.hidden = s.id !== "sec-" + name; });
  $("#section-title").textContent = SECTION_TITLES[name] || name;

  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  if (name === "dashboard") loadDashboard();
  if (name === "students") loadStudents();
  if (name === "cameras") loadCameras();
  if (name === "attendance") loadAttendance();
  if (name === "live") {
    loadLive();
    liveTimer = setInterval(loadLive, 5000);
    if (!ws || ws.readyState > 1) connectWS(); // CONNECTING/OPEN = 0/1
  }
  if (name === "system") loadSystem();
}

/* ------------------------------------------------------------- dashboard */
async function loadDashboard() {
  try {
    const h = await api("/system/health");
    const checks = h.checks || {};
    const up = Object.values(checks).filter((v) => v === "up" || v === "online").length;
    $("#st-health").textContent = h.status === "ok" ? "OK" : "Degraded";
    $("#st-health").className = "stat-value " + (h.status === "ok" ? "ok" : "warn");
    $("#st-health-sub").textContent = `${up}/${Object.keys(checks).length} checks up · v${h.version}`;
    $("#version-label").textContent = "v" + h.version;
  } catch (e) { $("#st-health").textContent = "?"; errToast(e); }

  try {
    const g = await api("/system/gpu-status");
    const on = g.service === "online";
    $("#st-inference").textContent = on ? "Online" : "Offline";
    $("#st-inference").className = "stat-value " + (on ? "ok" : "bad");
    $("#st-inference-sub").textContent = on
      ? `${g.provider || "?"} · ${g.device_name || ""}`
      : (g.detail || "not running");
    $("#dash-gpu").innerHTML = on
      ? kvHtml(g)
      : `<span class="muted">Inference service offline — start it (or Docker stack) to enable enrollment &amp; recognition.<br>Detail: <b>${esc(g.detail || "unknown")}</b></span>`;
  } catch (e) { $("#st-inference").textContent = "?"; }

  try {
    const s = await api("/students?limit=1");
    $("#st-students").textContent = s.total;
  } catch { $("#st-students").textContent = "?"; }

  try {
    const cams = await api("/cameras");
    $("#st-cameras").textContent = cams.length;
    const running = cams.filter((c) => c.runtime && c.runtime.state === "running").length;
    $("#st-cameras-sub").textContent = `${running} running`;
  } catch { $("#st-cameras").textContent = "?"; }

  try {
    const sum = await api(`/attendance/summary?date=${localToday()}`);
    $("#st-present").textContent = sum.totals.present_students;
    $("#st-present-sub").textContent =
      `${sum.totals.ongoing_sessions} ongoing · ${sum.totals.completed_sessions} completed`;
  } catch { $("#st-present").textContent = "?"; }
}

function kvHtml(obj) {
  const skip = new Set(["throughput"]);
  return Object.entries(obj)
    .filter(([k, v]) => !skip.has(k) && v !== null && v !== "")
    .map(([k, v]) => `<span>${esc(k)}: <b>${esc(Array.isArray(v) ? v.join(", ") : typeof v === "object" ? JSON.stringify(v) : v)}</b></span>`)
    .join("");
}

/* ------------------------------------------------------------- students */
let studentQuery = "";

async function loadStudents() {
  try {
    const data = await api(`/students?limit=200&q=${encodeURIComponent(studentQuery)}`);
    $("#students-count").textContent = `${data.total} total`;
    const body = $("#students-body");
    if (!data.items.length) {
      body.innerHTML = `<tr><td colspan="7" class="muted">No students yet — enroll your first one (3–5 photos).</td></tr>`;
      return;
    }
    body.innerHTML = data.items.map((s) => `
      <tr>
        <td>${s.id}</td>
        <td>${esc(s.name)}</td>
        <td>${esc(s.registration_no)}</td>
        <td>${esc(s.section || "—")}</td>
        <td>${(s.photo_paths || []).length}</td>
        <td>${s.embedding_count ? `<span class="badge green">${s.embedding_count}</span>` : `<span class="badge red">0</span>`}</td>
        <td><div class="row-actions">
          <button class="btn small" data-act="add-photos" data-id="${s.id}">+ photos</button>
          <button class="btn small danger" data-act="del-face" data-id="${s.id}">clear face</button>
          <button class="btn small danger" data-act="del" data-id="${s.id}">delete</button>
        </div></td>
      </tr>`).join("");
  } catch (e) { errToast(e); }
}

async function onStudentAction(ev) {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const id = btn.dataset.id, act = btn.dataset.act;
  try {
    if (act === "del") {
      if (!confirm(`Delete student #${id} and all their sessions?`)) return;
      await api(`/students/${id}`, { method: "DELETE" });
      toast("Student deleted", "ok");
      loadStudents();
    } else if (act === "del-face") {
      if (!confirm(`Clear face data for student #${id}?`)) return;
      await api(`/students/${id}/face`, { method: "DELETE" });
      toast("Face data cleared", "ok");
      loadStudents();
    } else if (act === "add-photos") {
      const input = $("#photo-input");
      input.onchange = async () => {
        if (!input.files.length) return;
        const fd = new FormData();
        for (const f of input.files) fd.append("photos", f);
        input.value = "";
        try {
          await api(`/students/${id}/photos`, { form: fd });
          toast(`Added ${fd.getAll("photos").length} photo(s)`, "ok");
          loadStudents();
        } catch (e) { errToast(e); }
      };
      input.click();
    }
  } catch (e) { errToast(e); }
}

async function submitEnrollment(ev) {
  ev.preventDefault();
  const files = $("#s-photos").files;
  if (files.length < 3 || files.length > 5) {
    toast("Enrollment needs 3–5 photos (selected: " + files.length + ")", "error");
    return;
  }
  const fd = new FormData();
  fd.append("name", $("#s-name").value.trim());
  fd.append("registration_no", $("#s-reg").value.trim());
  const section = $("#s-section").value.trim();
  if (section) fd.append("section", section);
  for (const f of files) fd.append("photos", f);

  const btn = $("#enroll-form button[type=submit]");
  btn.disabled = true;
  try {
    const s = await api("/students", { form: fd });
    toast(`Enrolled ${s.name} (${s.embedding_count} embeddings)`, "ok");
    $("#enroll-form").reset();
    $("#s-photos-count").textContent = "no files";
    $("#enroll-form").hidden = true;
    loadStudents();
  } catch (e) { errToast(e); }
  btn.disabled = false;
}

/* ------------------------------------------------------------- cameras */
function camStateBadge(c) {
  const st = (c.runtime && c.runtime.state) || "stopped";
  const cls = st === "running" ? "green" : st === "unhealthy" ? "red"
    : (st === "starting" || st === "stopping") ? "amber" : "gray";
  return `<span class="badge ${cls}">${esc(st)}</span>${c.runtime && c.runtime.slot != null ? ` <span class="muted small">#${c.runtime.slot}</span>` : ""}`;
}

async function loadCameras() {
  try {
    const cams = await api("/cameras");
    $("#cameras-count").textContent = `${cams.length} total`;
    const body = $("#cameras-body");
    if (!cams.length) {
      body.innerHTML = `<tr><td colspan="8" class="muted">No cameras yet — add an RTSP stream or a local video file.</td></tr>`;
      return;
    }
    body.innerHTML = cams.map((c) => {
      const src = c.rtsp_url ? c.rtsp_url.replace(/^rtsp:\/\//, "rtsp://").slice(0, 42)
        : c.file_path ? esc(c.file_path) : "—";
      const running = c.runtime && (c.runtime.state === "running" || c.runtime.state === "starting");
      return `<tr>
        <td>${c.id}</td>
        <td>${esc(c.name)}</td>
        <td>${esc(c.location || "—")}</td>
        <td><span class="badge gray">${esc(c.type)}</span></td>
        <td title="${esc(c.rtsp_url || c.file_path || "")}">${src}</td>
        <td>${c.sampling_rate}/s</td>
        <td>${camStateBadge(c)}</td>
        <td><div class="row-actions">
          ${running
            ? `<button class="btn small" data-act="stop" data-id="${c.id}">stop</button>`
            : `<button class="btn small primary" data-act="start" data-id="${c.id}">start</button>`}
          <button class="btn small danger" data-act="cam-del" data-id="${c.id}">delete</button>
        </div></td>
      </tr>`;
    }).join("");
  } catch (e) { errToast(e); }
}

async function onCameraAction(ev) {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const id = btn.dataset.id, act = btn.dataset.act;
  btn.disabled = true;
  try {
    if (act === "start") {
      const r = await api(`/cameras/${id}/start`, { method: "POST", body: {} });
      toast(`Camera #${id}: ${r.state}${r.slot != null ? " (slot " + r.slot + ")" : ""}`, "ok");
    } else if (act === "stop") {
      const r = await api(`/cameras/${id}/stop`, { method: "POST", body: {} });
      toast(`Camera #${id}: ${r.state}`, "ok");
    } else if (act === "cam-del") {
      if (!confirm(`Delete camera #${id}?`)) return;
      await api(`/cameras/${id}`, { method: "DELETE" });
      toast("Camera deleted", "ok");
    }
    loadCameras();
  } catch (e) { errToast(e); }
  btn.disabled = false;
}

async function submitCamera(ev) {
  ev.preventDefault();
  const rtsp = $("#c-rtsp").value.trim();
  const file = $("#c-file").value.trim();
  if (!rtsp && !file) { toast("Provide an RTSP URL or a video file path", "error"); return; }
  const body = {
    name: $("#c-name").value.trim(),
    type: $("#c-type").value,
    sampling_rate: Number($("#c-rate").value) || 5,
  };
  const loc = $("#c-location").value.trim();
  if (loc) body.location = loc;
  if (rtsp) body.rtsp_url = rtsp;
  if (file) body.file_path = file;
  try {
    const c = await api("/cameras", { body });
    toast(`Camera "${c.name}" created (#${c.id})`, "ok");
    $("#camera-form").reset();
    $("#c-rate").value = 5;
    $("#camera-form").hidden = true;
    loadCameras();
  } catch (e) { errToast(e); }
}

/* ------------------------------------------------------------- attendance */
async function loadAttendance() {
  const date = $("#att-date").value || localToday();
  const absent = $("#att-absent").checked ? "&include_absent=true" : "";
  try {
    const sum = await api(`/attendance/summary?date=${date}${absent}`);
    $("#att-present").textContent = sum.totals.present_students;
    $("#att-ongoing").textContent = sum.totals.ongoing_sessions;
    $("#att-completed").textContent = sum.totals.completed_sessions;
    $("#att-hours").textContent = fmtDur(sum.totals.total_seconds);
    $("#summary-count").textContent = `${sum.students.length} rows · ${date}`;
    const body = $("#summary-body");
    body.innerHTML = sum.students.length ? sum.students.map((r) => `
      <tr>
        <td>${esc(r.name)}</td>
        <td>${esc(r.registration_no)}</td>
        <td>${esc(r.section || "—")}</td>
        <td>${r.entries}</td>
        <td>${r.exits}</td>
        <td>${fmtDT(r.first_entry)}</td>
        <td>${fmtDT(r.last_exit)}</td>
        <td>${fmtDur(r.total_seconds)}</td>
        <td>${r.ongoing ? `<span class="badge green">inside</span>` : `<span class="badge gray">out</span>`}</td>
      </tr>`).join("")
      : `<tr><td colspan="9" class="muted">No attendance recorded for ${esc(date)}.</td></tr>`;
  } catch (e) { errToast(e); }

  try {
    const logs = await api("/attendance/logs?limit=200");
    const items = logs.items || [];
    $("#logs-count").textContent = `${items.length} of ${logs.total ?? "?"}`;
    $("#logs-body").innerHTML = items.length ? items.map((l) => `
      <tr>
        <td>${fmtDT(l.timestamp)}</td>
        <td>${esc(l.student_name || (l.student_id != null ? "#" + l.student_id : "—"))}</td>
        <td>${esc(l.camera_name || (l.camera_id != null ? "#" + l.camera_id : "—"))}</td>
        <td>${l.confidence_score != null ? Number(l.confidence_score).toFixed(3) : "—"}</td>
        <td>${l.gpu_inference_time_ms != null ? Number(l.gpu_inference_time_ms).toFixed(1) : "—"}</td>
      </tr>`).join("")
      : `<tr><td colspan="5" class="muted">No recognition events yet.</td></tr>`;
  } catch (e) { errToast(e); }
}

/* ------------------------------------------------------------- live */
async function loadLive() {
  try {
    const live = await api("/attendance/live");
    $("#live-count").textContent = `${live.count} ongoing`;
    $("#live-body").innerHTML = live.sessions.length ? live.sessions.map((s) => `
      <tr>
        <td>${esc(s.student && (s.student.name || s.student.registration_no)) || "#" + (s.session_id ?? "?")}</td>
        <td>${fmtDT(s.entry_time)}</td>
        <td><span class="badge green">${fmtDur(s.elapsed_seconds)}</span></td>
        <td>${esc(s.camera_in_name || "—")}</td>
      </tr>`).join("")
      : `<tr><td colspan="4" class="muted">Nobody inside right now.</td></tr>`;
  } catch (e) { /* avoid toast spam on 5s timer */ console.warn(e); }
}

let ws = null;
let wsRetry = null;

function connectWS() {
  const tok = getToken();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws/live-feed${tok ? "?token=" + encodeURIComponent(tok) : ""}`;
  try { ws = new WebSocket(url); } catch { return; }

  ws.onopen = () => {
    $("#ws-dot").className = "dot on";
    $("#ws-status").textContent = "connected";
  };
  ws.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch { ev = { raw: m.data }; }
    const box = $("#events");
    if (box.querySelector(".muted")) box.innerHTML = "";
    const div = document.createElement("div");
    div.className = "event";
    div.innerHTML = `<span class="ev-type">${esc(ev.type || ev.event || "event")}</span>
      <span class="ev-time">${new Date().toLocaleTimeString()}</span>
      <pre>${esc(JSON.stringify(ev, null, 2))}</pre>`;
    box.prepend(div);
    while (box.children.length > 60) box.lastChild.remove();
  };
  ws.onclose = () => {
    $("#ws-dot").className = "dot off";
    $("#ws-status").textContent = "disconnected — retrying…";
    clearTimeout(wsRetry);
    wsRetry = setTimeout(() => { if (getToken() !== null) connectWS(); }, 4000);
  };
  ws.onerror = () => { try { ws.close(); } catch {} };
}

/* ------------------------------------------------------------- system */
async function loadSystem() {
  try {
    const h = await api("/system/health");
    $("#health-body").textContent = JSON.stringify(h, null, 2);
  } catch (e) { $("#health-body").textContent = String(e.message || e); }
  try {
    const g = await api("/system/gpu-status");
    $("#gpu-body").textContent = JSON.stringify(g, null, 2);
  } catch (e) { $("#gpu-body").textContent = String(e.message || e); }
}

/* ------------------------------------------------------------- wiring */
document.addEventListener("DOMContentLoaded", () => {
  $("#login-form").addEventListener("submit", tryLogin);
  $("#btn-logout").addEventListener("click", logout);

  $$(".nav-btn").forEach((b) => b.addEventListener("click", () => showSection(b.dataset.section)));

  $("#btn-enroll-toggle").addEventListener("click", () => { $("#enroll-form").hidden = !$("#enroll-form").hidden; });
  $("#btn-enroll-cancel").addEventListener("click", () => { $("#enroll-form").hidden = true; });
  $("#enroll-form").addEventListener("submit", submitEnrollment);
  $("#s-photos").addEventListener("change", () => {
    const n = $("#s-photos").files.length;
    $("#s-photos-count").textContent = n ? `${n} file(s) selected` : "no files";
  });
  $("#students-body").addEventListener("click", onStudentAction);
  $("#stu-q").addEventListener("input", (e) => {
    studentQuery = e.target.value.trim();
    clearTimeout(window.__stuTimer);
    window.__stuTimer = setTimeout(loadStudents, 300);
  });

  $("#btn-camera-toggle").addEventListener("click", () => { $("#camera-form").hidden = !$("#camera-form").hidden; });
  $("#btn-camera-cancel").addEventListener("click", () => { $("#camera-form").hidden = true; });
  $("#camera-form").addEventListener("submit", submitCamera);
  $("#cameras-body").addEventListener("click", onCameraAction);

  $("#att-date").value = localToday();
  $("#btn-att-refresh").addEventListener("click", loadAttendance);
  $("#att-date").addEventListener("change", loadAttendance);
  $("#att-absent").addEventListener("change", loadAttendance);

  $("#btn-live-refresh").addEventListener("click", loadLive);
  $("#btn-sys-refresh").addEventListener("click", loadSystem);

  boot();
});
