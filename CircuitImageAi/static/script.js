class DashboardController {
    constructor() {
        this.currentMode = null;
        this.lastDesignData = null;
        this.bodeChartInstance = null;
        this.trainingES = null;

        this.bindInlineHandlers();
        this.bindPageEvents();
        this.loadVowels();
    }

    bindInlineHandlers() {
        [
            "showPanel",
            "startTraining",
            "resumeTraining",
            "predict",
            "computeDesignRC",
            "loadVowelData",
            "downloadNetlist",
            "updateTopologyImage",
            "loadDefaultModel",
            "uploadModel",
            "logout",
        ].forEach((methodName) => {
            window[methodName] = this[methodName].bind(this);
        });
    }

    bindPageEvents() {
        document.getElementById("known-select").addEventListener("change", (event) => {
            const known = event.currentTarget.value;
            document.getElementById("r-input").disabled = known !== "R";
            document.getElementById("c-input").disabled = known !== "C";
        });
    }

    async loadVowels() {
        const response = await fetch("/get_vowels_list");
        const data = await response.json();
        const select = document.getElementById("vowel-select");

        if (select && data.vowels) {
            data.vowels.forEach((vowel) => {
                const option = document.createElement("option");
                option.value = vowel;
                option.innerText = vowel;
                select.appendChild(option);
            });
        }
    }

    showPanel(panelName) {
        const mainDashboard = document.getElementById("main-dashboard-view");
        const settings = document.getElementById("settings-panel");
        const designPanel = document.getElementById("unified-design-panel");

        if (mainDashboard) mainDashboard.style.display = "none";
        if (settings) settings.style.display = "none";
        if (designPanel) designPanel.style.display = "none";

        if (panelName === "settings" && settings) {
            settings.style.display = "block";
        } else if (panelName === "dashboard" && mainDashboard) {
            mainDashboard.style.display = "block";
        }
    }

    startTraining() {
        this.startTrainingStream("/train_stream", "Starting training stream...\n");
    }

    resumeTraining() {
        this.startTrainingStream("/train_stream_resume", "Resuming training...\n");
    }

    startTrainingStream(endpoint, initialMessage) {
        this.showPanel("dashboard");
        const log = document.getElementById("training-log");
        const progressBar = document.getElementById("progress-bar");

        if (!log) {
            console.error("training-log not found");
            return;
        }

        log.innerHTML = initialMessage;
        if (progressBar) progressBar.style.width = "0%";

        if (this.trainingES) this.trainingES.close();
        this.trainingES = new EventSource(endpoint);
        this.trainingES.onmessage = (event) => {
            log.innerHTML += event.data + "<br>";
            log.scrollTop = log.scrollHeight;

            const progress = event.data.match(/Progress:\s*(\d+)%/);
            if (progress && progressBar) progressBar.style.width = progress[1] + "%";

            if (event.data.includes("Training completed")) {
                this.trainingES.close();
                if (progressBar) progressBar.style.width = "100%";
            }
        };
    }

    async predict() {
        this.showPanel("dashboard");

        const fileInput = document.getElementById("fileInput");
        if (!fileInput.files.length) {
            alert("Choose image");
            return;
        }

        const prediction = document.getElementById("prediction");
        prediction.innerHTML = "Processing...";
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        try {
            const response = await fetch("/predict", { method: "POST", body: formData });
            const data = await response.json();

            if (data.error) {
                prediction.innerHTML = data.error;
                return;
            }

            prediction.style.background = data.color;
            prediction.style.color = "#fff";
            prediction.style.padding = "10px";
            prediction.innerHTML = `<strong>${data.friendly}</strong> (${data.confidence}%)`;
            this.setupUI(data.class);
        } catch (error) {
            console.error(error);
        }
    }

    setupUI(circuitClass) {
        this.currentMode = circuitClass;

        const mainDashboard = document.getElementById("main-dashboard-view");
        const panel = document.getElementById("unified-design-panel");
        const rcInputs = document.getElementById("rc-inputs");
        const resonatorInputs = document.getElementById("res-inputs");
        const title = document.getElementById("design-title");

        if (mainDashboard) mainDashboard.style.display = "block";
        document.getElementById("settings-panel").style.display = "none";
        if (!panel) return;

        panel.style.display = "none";
        rcInputs.style.display = "none";
        resonatorInputs.style.display = "none";
        document.getElementById("results-area").style.display = "none";

        if (circuitClass === "rc_lp" || circuitClass === "rc_hp") {
            panel.style.display = "block";
            rcInputs.style.display = "block";
            title.innerText = (circuitClass === "rc_lp" ? "Lowpass" : "Highpass") + " Designer";
        } else if (circuitClass === "resonator") {
            panel.style.display = "block";
            resonatorInputs.style.display = "block";
            title.innerText = "Formant Resonator Designer (Pre-Calculated)";
        }
    }

    async loadVowelData() {
        const vowel = document.getElementById("vowel-select").value;
        if (!vowel) {
            alert("Please select a vowel first.");
            return;
        }

        const response = await fetch(`/get_vowel_design/${vowel}`);
        const data = await response.json();
        if (data.error) {
            alert(data.error);
            return;
        }
        this.showResults(data);
    }

    async computeDesignRC() {
        const payload = {
            rc_type: this.currentMode,
            fc_hz: document.getElementById("fc-input").value,
            known: document.getElementById("known-select").value,
            R_ohm: parseFloat(document.getElementById("r-input").value),
            C_f: parseFloat(document.getElementById("c-input").value),
        };
        const response = await fetch("/design_rc", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }
        this.showResults(data);
    }

    showResults(data) {
        this.lastDesignData = data.design;
        document.getElementById("results-area").style.display = "block";
        const textResults = document.getElementById("text-results");
        let html = "";

        if (data.components) {
            html += `<h4 style="margin-bottom:10px; border-bottom:1px solid #ddd; padding-bottom:5px;">Vowel Circuit Parameters</h4>`;
            html += `<ul style="list-style:none; padding:0;">`;
            data.components.forEach((component) => {
                html += `<li style="margin-bottom:10px; background:#fff; padding:8px; border-radius:6px; border:1px solid #eee;">
                            <strong>Stage ${component.id}:</strong> F_peak=${component.peak}Hz, BW=${component.bw}Hz <br>
                            <span style="color:#e67e22; font-size:0.9em;">(Natural Freq f0=${component.f0}Hz, Q=${component.Q})</span><br>
                            <strong>R1:</strong> ${component.R1}, <strong>R2:</strong> ${component.R2}<br>
                            <strong>C1:</strong> ${component.C1}, <strong>C2:</strong> ${component.C2}
                         </li>`;
            });
            html += "</ul>";
        } else if (data.display) {
            html += `<h4>RC Filter Results</h4>`;
            html += `<strong>R:</strong> ${data.display.R}<br><strong>C:</strong> ${data.display.C}`;
        }

        textResults.innerHTML = html;
        this.drawBode(data.bode.freqs, data.bode.mags);
    }

    drawBode(frequencies, magnitudes) {
        const context = document.getElementById("bodeChart").getContext("2d");
        if (this.bodeChartInstance) this.bodeChartInstance.destroy();

        this.bodeChartInstance = new Chart(context, {
            type: "line",
            data: {
                labels: frequencies.map((frequency) => frequency.toExponential(1)),
                datasets: [{
                    label: "Gain (dB)",
                    data: magnitudes,
                    borderColor: "#e74c3c",
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                plugins: {
                    title: { display: true, text: "Frequency Response (Bode Plot)" },
                },
                scales: {
                    x: { title: { display: true, text: "Freq (Hz)" }, ticks: { maxTicksLimit: 8 } },
                    y: { title: { display: true, text: "Magnitude (dB)" } },
                },
            },
        });
    }

    updateTopologyImage() {
        const topology = document.getElementById("topology-select").value;
        const image = document.getElementById("topology-img");
        if (topology) {
            image.src = `/static/${topology}.png`;
            image.style.display = "inline-block";
        } else {
            image.style.display = "none";
        }
    }

    async downloadNetlist() {
        if (!this.lastDesignData) return;

        if (this.currentMode === "resonator") {
            const topology = document.getElementById("topology-select").value;
            if (!topology) {
                alert("Please select a Resonator Topology before downloading the netlist.");
                return;
            }
            this.lastDesignData.topology = topology;
        }

        const format = document.getElementById("nl-format").value;
        const response = await fetch("/export_netlist", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                format,
                design_data: this.lastDesignData,
                lib: document.getElementById("nl-lib").value,
                cell: document.getElementById("nl-cell").value,
                view: document.getElementById("nl-view").value,
            }),
        });

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;

        const extensions = { spectre: "scs", kicad: "net", ltspice: "cir", spice: "cir" };
        const disposition = response.headers.get("Content-Disposition") || "";
        const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1];
        link.download = filename || `netlist.${extensions[format] || "cir"}`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => window.URL.revokeObjectURL(url), 1000);
    }

    async loadDefaultModel() {
        const response = await fetch("/load_model_default", { method: "POST" });
        const data = await response.json();
        if (data.status === "ok") {
            document.getElementById("model-status").innerText = "Loaded: " + data.classes.join(", ");
        } else {
            alert(data.message);
        }
    }

    async uploadModel() {
        const file = document.getElementById("modelFile").files[0];
        if (!file) {
            alert("Select file");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("/load_model_file", { method: "POST", body: formData });
        const data = await response.json();
        if (data.status === "ok") {
            document.getElementById("model-status").innerText = "Loaded: " + data.classes.join(", ");
        } else {
            alert(data.message);
        }
    }

    logout() {
        window.location.href = "/logout";
    }
}

window.addEventListener("DOMContentLoaded", () => {
    window.dashboardController = new DashboardController();
});
