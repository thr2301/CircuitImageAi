function startTraining() {
    const output = document.getElementById("output");
    const progressBar = document.getElementById("progress-bar");
    output.innerText = "";
    progressBar.style.width = "0%";

    const evtSource = new EventSource("/train_stream");
    evtSource.onmessage = function(event) {
        output.innerText += event.data + "\n";
        output.scrollTop = output.scrollHeight;

        const match = event.data.match(/Progress: (\d+)/);
        if(match) progressBar.style.width = match[1] + "%";

        if(event.data.includes("completed")) {
            progressBar.style.width = "100%";
            evtSource.close();
        }
    };
}

async function predict() {
    const fileInput = document.getElementById("fileInput");
    if(!fileInput.files.length){ alert("Please choose an image."); return; }

    const predDiv = document.getElementById("prediction");
    predDiv.innerText = "Predicting...";
    predDiv.style.backgroundColor = "#34495e";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const res = await fetch("/predict",{method:"POST",body:formData});
        const data = await res.json();

        if(data.error) {
            predDiv.innerText = data.error;
            predDiv.style.backgroundColor = "#e74c3c";
        } else {
            predDiv.innerText = data.friendly || "Unknown";
            predDiv.style.backgroundColor = data.color || "#000";
        }
    } catch(err) {
        predDiv.innerText = "Prediction failed!";
        predDiv.style.backgroundColor = "#e74c3c";
        console.error(err);
    }
}

function logout(){ window.location.href = "/"; }
