let lastPredClass = null; // "rc_lp", "rc_hp", "amplifier", "other"
let trainingES = null;

function startTraining() {
  const logEl = document.getElementById("training-log");
  const trainBtn = document.getElementById("train-btn"); // add id to your Train button
  const progressBar = document.getElementById("progress-bar"); // optional

  if (!logEl) {
    console.error("#training-log not found in DOM.");
    return;
  }

  // reset UI
  logEl.innerHTML = "";
  if (progressBar) progressBar.style.width = "0%";

  // close any previous stream
  if (trainingES) {
    try { trainingES.close(); } catch (_) {}
    trainingES = null;
  }

  // disable button during training
  if (trainBtn) trainBtn.disabled = true;

  // open SSE
  trainingES = new EventSource("/train_stream");

  trainingES.onmessage = function (event) {
    const msg = event.data || "";
    logEl.innerHTML += msg + "<br>";
    logEl.scrollTop = logEl.scrollHeight;

    // optional progress parsing: expects "Progress: NN%"
    const m = msg.match(/Progress:\s*(\d+)%?/i);
    if (m && progressBar) progressBar.style.width = `${m[1]}%`;

    if (msg.includes("Training completed")) {
      if (progressBar) progressBar.style.width = "100%";
      trainingES.close();
      trainingES = null;
      if (trainBtn) trainBtn.disabled = false;
    }
  };

  trainingES.onerror = function () {
    logEl.innerHTML += "<span style='color:red;'>Error in training stream.</span><br>";
    try { trainingES.close(); } catch (_) {}
    trainingES = null;
    if (trainBtn) trainBtn.disabled = false;
  };
}

function logout(){ window.location.href = "/"; }

async function predict() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files.length) { alert("Please choose an image."); return; }

  const predDiv = document.getElementById("prediction");
  predDiv.innerText = "Predicting...";
  predDiv.style.backgroundColor = "#34495e";
  predDiv.style.color = "#fff";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      predDiv.innerText = data.error;
      predDiv.style.backgroundColor = "#e74c3c";
      document.getElementById("rc-designer").style.display = "none";
      lastPredClass = null;
      return;
    }

    const pct = (data.confidence ?? 0).toFixed(2);
    predDiv.style.backgroundColor = data.color || "#000";
    predDiv.style.color = "#fff";
    predDiv.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-size:18px;font-weight:700;">${data.friendly}</span>
        <span style="font-size:16px;">${pct}%</span>
      </div>
      <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
    `;

    lastPredClass = data.class; // "rc_lp", "rc_hp", "amplifier", "other"

    const designer = document.getElementById("rc-designer");
    const chip = document.getElementById("rc-type-chip");
    if (lastPredClass === "rc_lp" || lastPredClass === "rc_hp") {
      chip.textContent = (lastPredClass === "rc_lp") ? "RC Lowpass" : "RC Highpass";
      designer.style.display = "block";
    } else {
      designer.style.display = "none";
    }
  } catch (err) {
    predDiv.innerText = "Prediction failed!";
    predDiv.style.backgroundColor = "#e74c3c";
    console.error(err);
    document.getElementById("rc-designer").style.display = "none";
    lastPredClass = null;
  }
}

// Enable/disable R/C fields based on selection
document.addEventListener("DOMContentLoaded", () => {
  const knownSel = document.getElementById("known-select");
  const rInput = document.getElementById("r-input");
  const cInput = document.getElementById("c-input");

  if (knownSel) {
    knownSel.addEventListener("change", () => {
      const val = knownSel.value;
      if (val === "R") {
        rInput.disabled = false;
        cInput.disabled = true; cInput.value = "";
      } else if (val === "C") {
        rInput.disabled = true; rInput.value = "";
        cInput.disabled = false;
      } else {
        rInput.disabled = true; rInput.value = "";
        cInput.disabled = true; cInput.value = "";
      }
    });
  }
});

// Keep last computed RC values for export
let lastRC = null;

async function computeRC() {
  if (!(lastPredClass === "rc_lp" || lastPredClass === "rc_hp")) {
    alert("RC Designer is only available when the prediction is an RC filter.");
    return;
  }

  const fcEl = document.getElementById("fc-input");
  const knownSel = document.getElementById("known-select");
  const rEl = document.getElementById("r-input");
  const cEl = document.getElementById("c-input");
  const qEl = document.getElementById("q-input");
  const resultEl = document.getElementById("rc-result");
  const dlBtn = document.getElementById("dl-netlist-btn");

  const fc = parseFloat(fcEl.value);
  if (!fc || fc <= 0) {
    resultEl.style.display = "block";
    resultEl.innerHTML = `<span style="color:#e74c3c;">Please enter a valid cutoff frequency (Hz).</span>`;
    dlBtn.disabled = true;
    lastRC = null;
    return;
  }

  const payload = {
    rc_type: lastPredClass,            // "rc_lp" or "rc_hp"
    fc_hz: fc,
    known: knownSel.value,
    R_ohm: rEl.disabled ? null : (rEl.value ? parseFloat(rEl.value) : null),
    C_f: cEl.disabled ? null : (cEl.value ? parseFloat(cEl.value) : null),
    Q: qEl.value ? parseFloat(qEl.value) : null
  };

  try {
    const res = await fetch("/design_rc", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    resultEl.style.display = "block";
    if (data.error) {
      resultEl.innerHTML = `<span style="color:#e74c3c;">${data.error}</span>`;
      dlBtn.disabled = true;
      lastRC = null;
      return;
    }

    const note = data.note ? `<div style="margin-top:6px; opacity:.8;">${data.note}</div>` : "";
    resultEl.innerHTML = `
      <div><strong>Cutoff:</strong> ${data.fc_hz} Hz</div>
      <div><strong>R:</strong> ${data.R_pretty} (${data.R_ohm.toPrecision(4)} Ω)</div>
      <div><strong>C:</strong> ${data.C_pretty} (${data.C_f.toExponential(3)} F)</div>
      ${note}
    `;

    // Store for export and enable button
    lastRC = {
      rc_type: data.rc_type,
      R_ohm: data.R_ohm,
      C_f: data.C_f
    };
    dlBtn.disabled = false;

  } catch (err) {
    resultEl.style.display = "block";
    resultEl.innerHTML = `<span style="color:#e74c3c;">Design failed. Check inputs and try again.</span>`;
    console.error(err);
    dlBtn.disabled = true;
    lastRC = null;
  }
}

async function downloadNetlist() {
  if (!lastRC) { alert("Compute R & C first."); return; }

  const defaultName = `${lastRC.rc_type}.scs`;
  const filename = prompt("File name:", defaultName) || defaultName;

  const lib  = (document.getElementById("nl-lib")?.value || "").trim();
  const cell = (document.getElementById("nl-cell")?.value || "").trim();
  const view = (document.getElementById("nl-view")?.value || "schematic").trim();

  if (!lib || !cell) {
    alert("For Spectre export, please provide both Library and Cell names.");
    return;
  }

  const payload = {
    rc_type: lastRC.rc_type,
    R_ohm: lastRC.R_ohm,
    C_f: lastRC.C_f,
    title: `Auto netlist for ${lastRC.rc_type}`,
    ac_start: 1,
    ac_stop: 1e6,
    ac_pts: 200,
    filename,
    lib,
    cell,
    view
  };

  const res = await fetch("/export_netlist", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const txt = await res.text();
    alert("Failed to generate netlist: " + txt);
    return;
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function loadDefaultModel() {
  const status = document.getElementById("model-status");
  status.textContent = "Loading best_model.pth...";
  try {
    const res = await fetch("/load_model_default", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed");
    status.textContent = `✅ Loaded. Classes: ${data.classes.join(", ")}`;
  } catch (e) {
    status.textContent = "❌ " + e.message;
  }
}

async function uploadModel() {
  const status = document.getElementById("model-status");
  const fileInput = document.getElementById("modelFile");
  if (!fileInput.files.length) {
    alert("Choose a .pth file first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  status.textContent = "Uploading and loading model...";
  try {
    const res = await fetch("/load_model_file", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed");
    status.textContent = `✅ ${data.message}. Classes: ${data.classes.join(", ")}`;
  } catch (e) {
    status.textContent = "❌ " + e.message;
  }
}

async function registerUser(event) {
  event.preventDefault();
  const form = document.getElementById("registerForm");
  const formData = new FormData(form);

  try {
    const res = await fetch("/register", { method: "POST", body: formData });

    // if your API sometimes returns HTML (e.g., error template), guard parsing:
    const contentType = res.headers.get("content-type") || "";
    const data = contentType.includes("application/json") ? await res.json() : {};

    if (!res.ok || data.error) {
      const msg = (data && data.error) ? data.error : "Registration failed.";
      document.getElementById("register-msg").innerText = msg;
      document.getElementById("register-msg").style.color = "red";
      return;
    }

    // SUCCESS: choose ONE of these

    // A) Hard redirect to login page (your login is at "/")
    window.location.replace("/?registered=1");

    // B) Or, if you prefer your SPA panel switch:
    // document.getElementById("register-msg").innerText = data.message || "Registration successful";
    // document.getElementById("register-msg").style.color = "green";
    // setTimeout(() => toggleAuth("login"), 800);

  } catch (err) {
    console.error(err);
    document.getElementById("register-msg").innerText = "Registration failed.";
    document.getElementById("register-msg").style.color = "red";
  }
}

document.getElementById("registerForm")?.addEventListener("submit", registerUser);

