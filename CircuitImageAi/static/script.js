let currentMode = null; // "rc_lp", "rc_hp", "resonator"
let lastDesignData = null; 
let bodeChartInstance = null;
let trainingES = null;

// --- On Page Load: Fetch Vowel List ---
window.addEventListener('DOMContentLoaded', () => {
    fetch("/get_vowels_list")
    .then(r => r.json())
    .then(data => {
        const sel = document.getElementById("vowel-select");
        if(sel && data.vowels) {
            data.vowels.forEach(v => {
                let opt = document.createElement("option");
                opt.value = v;
                opt.innerText = v;
                sel.appendChild(opt);
            });
        }
    });
});

// --- Panel Switching Logic ---
function showPanel(panelName) {
    const mainDash = document.getElementById("main-dashboard-view");
    const settings = document.getElementById("settings-panel");
    const designPanel = document.getElementById("unified-design-panel");
    
    // Hide everything safely
    if(mainDash) mainDash.style.display = "none";
    if(settings) settings.style.display = "none";
    if(designPanel) designPanel.style.display = "none";

    // Show requested
    if (panelName === 'settings') {
        if(settings) settings.style.display = "block";
    } 
    else if (panelName === 'dashboard') {
        if(mainDash) mainDash.style.display = "block";
    }
}

// --- Training Stream Logic ---
function startTraining() {
  showPanel('dashboard');
  const logEl = document.getElementById("training-log");
  const bar = document.getElementById("progress-bar");
  
  if(!logEl) { console.error("training-log not found"); return; }
  
  logEl.innerHTML = "Starting training stream...\n";
  if(bar) bar.style.width = "0%";
  
  if (trainingES) trainingES.close();
  trainingES = new EventSource("/train_stream");
  
  trainingES.onmessage = function(e) {
    logEl.innerHTML += e.data + "<br>";
    logEl.scrollTop = logEl.scrollHeight;
    
    const m = e.data.match(/Progress:\s*(\d+)%/);
    if(m && bar) bar.style.width = m[1] + "%";
    
    if(e.data.includes("Training completed")) {
        trainingES.close();
        if(bar) bar.style.width = "100%";
    }
  };
}

function resumeTraining() {
  showPanel('dashboard');
  const logEl = document.getElementById("training-log");
  const bar = document.getElementById("progress-bar");
  
  logEl.innerHTML = "Resuming training...\n";
  if(bar) bar.style.width = "0%";
  
  if (trainingES) trainingES.close();
  trainingES = new EventSource("/train_stream_resume");
  
  trainingES.onmessage = function(e) {
    logEl.innerHTML += e.data + "<br>";
    logEl.scrollTop = logEl.scrollHeight;
    
    const m = e.data.match(/Progress:\s*(\d+)%/);
    if(m && bar) bar.style.width = m[1] + "%";
    
    if(e.data.includes("Training completed")) {
        trainingES.close();
        if(bar) bar.style.width = "100%";
    }
  };
}

// --- Prediction Logic ---
async function predict() {
  showPanel('dashboard');
  
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files.length) { alert("Choose image"); return; }
  
  const predDiv = document.getElementById("prediction");
  predDiv.innerHTML = "Processing...";
  
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  
  try {
    const res = await fetch("/predict", {method: "POST", body: fd});
    const data = await res.json();
    
    if(data.error) { predDiv.innerHTML = data.error; return; }
    
    predDiv.style.background = data.color;
    predDiv.style.color = "#fff";
    predDiv.style.padding = "10px";
    predDiv.innerHTML = `<strong>${data.friendly}</strong> (${data.confidence}%)`;
    
    setupUI(data.class);
    
  } catch(e) { console.error(e); }
}

// --- Setup UI for Design Results ---
function setupUI(cls) {
    currentMode = cls;
    
    const mainDash = document.getElementById("main-dashboard-view");
    const panel = document.getElementById("unified-design-panel");
    const rcIn = document.getElementById("rc-inputs");
    const resIn = document.getElementById("res-inputs");
    const title = document.getElementById("design-title");

    // Ensure Dashboard is visible
    if(mainDash) mainDash.style.display = "block";
    document.getElementById("settings-panel").style.display = "none";

    if(panel) {
        panel.style.display = "none";
        rcIn.style.display = "none";
        resIn.style.display = "none";
        document.getElementById("results-area").style.display = "none";

        if(cls === "rc_lp" || cls === "rc_hp") {
            panel.style.display = "block";
            rcIn.style.display = "block";
            title.innerText = (cls==="rc_lp" ? "Lowpass" : "Highpass") + " Designer";
        } else if (cls === "resonator") {
            panel.style.display = "block";
            resIn.style.display = "block";
            title.innerText = "Formant Resonator Designer (Pre-Calculated)";
        }
    }
}

// --- Load Vowel Data ---
async function loadVowelData() {
    const vowel = document.getElementById("vowel-select").value;
    if (!vowel) { alert("Please select a vowel first."); return; }
    
    const res = await fetch(`/get_vowel_design/${vowel}`);
    const data = await res.json();
    
    if(data.error) { alert(data.error); return; }
    showResults(data);
}

// --- Compute RC Design ---
async function computeDesignRC() {
    const payload = {
        rc_type: currentMode,
        fc_hz: document.getElementById("fc-input").value,
        known: document.getElementById("known-select").value,
        R_ohm: parseFloat(document.getElementById("r-input").value),
        C_f: parseFloat(document.getElementById("c-input").value)
    };
    
    const res = await fetch("/design_rc", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    
    if(data.error) { alert(data.error); return; }
    showResults(data);
}

// --- Display Results & Bode Plot ---
function showResults(data) {
    lastDesignData = data.design; 
    document.getElementById("results-area").style.display = "block";
    const txtDiv = document.getElementById("text-results");
    
    let html = "";
    if (data.components) { // Resonator Vowel Data
        html += `<h4 style="margin-bottom:10px; border-bottom:1px solid #ddd; padding-bottom:5px;">Vowel Circuit Parameters</h4>`;
        html += `<ul style="list-style:none; padding:0;">`;
        data.components.forEach(c => {
            html += `<li style="margin-bottom:10px; background:#fff; padding:8px; border-radius:6px; border:1px solid #eee;">
                        <strong>Stage ${c.id}:</strong> F_peak=${c.peak}Hz, BW=${c.bw}Hz <br>
                        <span style="color:#e67e22; font-size:0.9em;">(Natural Freq f0=${c.f0}Hz, Q=${c.Q})</span><br>
                        <strong>R1:</strong> ${c.R1}, <strong>R2:</strong> ${c.R2}<br>
                        <strong>C1:</strong> ${c.C1}, <strong>C2:</strong> ${c.C2}
                     </li>`;
        });
        html += "</ul>";
    } else if (data.display) { // RC Data
        html += `<h4>RC Filter Results</h4>`;
        html += `<strong>R:</strong> ${data.display.R}<br><strong>C:</strong> ${data.display.C}`;
    }
    txtDiv.innerHTML = html;
    
    drawBode(data.bode.freqs, data.bode.mags);
}

// --- Chart.js Bode Plot ---
function drawBode(freqs, mags) {
    const ctx = document.getElementById('bodeChart').getContext('2d');
    if(bodeChartInstance) bodeChartInstance.destroy();
    
    bodeChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: freqs.map(f => f.toExponential(1)), 
            datasets: [{
                label: 'Gain (dB)',
                data: mags,
                borderColor: '#e74c3c',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: true, text: 'Frequency Response (Bode Plot)' }
            },
            scales: {
                x: { 
                    title: {display:true, text:'Freq (Hz)'}, 
                    ticks:{maxTicksLimit:8} 
                },
                y: { 
                    title: {display:true, text:'Magnitude (dB)'} 
                }
            }
        }
    });
}

// --- Download Netlist ---
async function downloadNetlist() {
    if(!lastDesignData) return;
    const fmt = document.getElementById("nl-format").value;
    const lib = document.getElementById("nl-lib").value;
    const cell = document.getElementById("nl-cell").value;
    const view = document.getElementById("nl-view").value;
    
    const res = await fetch("/export_netlist", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ 
            format: fmt, 
            design_data: lastDesignData,
            lib: lib, cell: cell, view: view 
        })
    });
    
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `netlist.${fmt === 'spectre' ? 'scs' : 'cir'}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
}

// --- Helper UI Listeners ---
document.getElementById("known-select").addEventListener("change", function(){
    const val = this.value;
    document.getElementById("r-input").disabled = (val !== "R");
    document.getElementById("c-input").disabled = (val !== "C");
});

// --- Model Handlers ---
function loadDefaultModel() {
    fetch("/load_model_default", {method:"POST"})
    .then(r=>r.json())
    .then(d=>{
        if(d.status==="ok") document.getElementById("model-status").innerText = "Loaded: " + d.classes.join(", ");
        else alert(d.message);
    });
}

function uploadModel() {
    const file = document.getElementById("modelFile").files[0];
    if(!file) return alert("Select file");
    const fd = new FormData();
    fd.append("file", file);
    fetch("/load_model_file", {method: "POST", body: fd})
    .then(r=>r.json())
    .then(d=>{
        if(d.status==="ok") document.getElementById("model-status").innerText = "Loaded: " + d.classes.join(", ");
        else alert(d.message);
    });
}

function logout() { window.location.href="/logout"; }