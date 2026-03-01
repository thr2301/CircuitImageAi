# app.py
import os
import time
import math
import datetime
import traceback
import json
from io import BytesIO
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, UploadFile, File, Body, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Torch / Vision ---
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

# --- Password hashing ---
from passlib.hash import bcrypt

# --- Math for Bode ---
import numpy as np

# =========================================================
# Configuration & Globals
# =========================================================
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
USERS_FILE = "users.json"
VOWELS_FILE = "vowel_results.txt"

# Cache for loaded vowel data
VOWEL_DATA_CACHE = {}

# --- 1. Model Definition (SimpleCNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class MLState:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ml_state = MLState()

# --- Helper: Load Vowel Data ---
def load_vowel_data():
    global VOWEL_DATA_CACHE
    if not os.path.exists(VOWELS_FILE):
        print(f"Warning: {VOWELS_FILE} not found. Please upload it.")
        return

    try:
        with open(VOWELS_FILE, "r", encoding="utf-8") as f:
            VOWEL_DATA_CACHE = json.load(f)
    except Exception as e:
        print(f"Error loading {VOWELS_FILE}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load Vowel Data
    load_vowel_data()
    
    # 2. Load ML Model
    default_path = "best_model.pth"
    
    # Try to infer classes from dataset
    data_dir = os.path.join("dataset", "train")
    if os.path.exists(data_dir):
        try:
            ds = datasets.ImageFolder(data_dir)
            ml_state.class_names = ds.classes
        except: pass

    if os.path.exists(default_path) and ml_state.class_names:
        try:
            ml_state.model = SimpleCNN(num_classes=len(ml_state.class_names)).to(ml_state.device)
            state_dict = torch.load(default_path, map_location=ml_state.device)
            ml_state.model.load_state_dict(state_dict)
            ml_state.model.eval()
        except Exception as e:
            print(f"Failed to load model: {e}")
    yield
    ml_state.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =========================================================
# Helpers
# =========================================================
def ensure_users_file_exists():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", encoding="utf-8") as f: json.dump([], f)
try: ensure_users_file_exists()
except: pass

def load_users() -> List[Dict]:
    ensure_users_file_exists()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except: return []

def save_users(users: List[Dict]):
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(users, f, indent=2)
    os.replace(tmp, USERS_FILE)

def authenticate_user(username, password):
    users = load_users()
    for u in users:
        if u.get("username") == username:
            try: return bcrypt.verify(password, u.get("password", ""))
            except: return False
    return False

def eng_suffix(value: float, kind: str) -> str:
    if isinstance(value, str):
        try: val = float(value)
        except: return value
    else:
        val = float(value)
        
    if val == 0: return "0"
    if kind == "R":
        if val >= 1e9: return f"{val/1e9:.3g}G"
        if val >= 1e6: return f"{val/1e6:.3g}M"
        if val >= 1e3: return f"{val/1e3:.3g}k"
    if kind == "C":
        if val < 1e-9: 
            if val < 1e-12: return f"{val/1e-15:.3g}f"
            return f"{val/1e-12:.3g}p"
        if val < 1e-6: return f"{val/1e-9:.3g}n"
        if val < 1e-3: return f"{val/1e-6:.3g}u"
        if val < 1:    return f"{val/1e-3:.3g}m"
    return f"{val:.4g}"

# --- Bode Helper ---
def calculate_bode(frequencies, H_s_func):
    s = 1j * 2 * np.pi * frequencies
    H = H_s_func(s)
    mag = np.abs(H)
    mag = np.where(mag < 1e-15, 1e-15, mag)
    mag_db = 20 * np.log10(mag)
    return mag_db.tolist()

# --- Netlist Generator ---
def make_netlist(fmt, design_data, lib="", cell="", view="schematic"):
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    d_type = design_data.get("type")
    title = f"{d_type} {now}"
    
    lines = []
    
    if fmt == "spectre":
        lines = [
            f"// Library name: {lib}",
            f"// Cell name: {cell}",
            f"// View name: {view}",
            f"// {title}",
            "simulator lang=spectre", "global 0",
            "Vsrc (in 0) vsource dc=0 mag=1 type=sine"
        ]
        
        if d_type in ["rc_lp", "rc_hp"]:
            R = eng_suffix(design_data["R"], "R")
            C = eng_suffix(design_data["C"], "C")
            if d_type == "rc_lp":
                lines.append(f"R1 (in out) resistor r={R}")
                lines.append(f"C1 (out 0) capacitor c={C}")
            else:
                lines.append(f"C1 (in out) capacitor c={C}")
                lines.append(f"R1 (out 0) resistor r={R}")
            lines.append("save v(out)")
            
        elif d_type == "resonator":
            stages = design_data.get("stages", [])
            for i, st in enumerate(stages):
                idx = i + 1
                suffix = f"_{idx}"
                R1 = eng_suffix(st["R1"], "R")
                R2 = eng_suffix(st["R2"], "R")
                C1 = eng_suffix(st["C1"], "C")
                C2 = eng_suffix(st["C2"], "C")
                f0 = float(st["f0"])
                Q = float(st["Q"])
                
                lines.append(f"// Stage {idx}: f0={f0:.1f}Hz, Q={Q:.2f}")
                lines.append(f"E{idx} (out{suffix} 0) vcvs pos=(n2{suffix} 0) gain=1.0")
                lines.append(f"R1{suffix} (in n1{suffix}) resistor r={R1}")
                lines.append(f"R2{suffix} (n1{suffix} n2{suffix}) resistor r={R2}")
                lines.append(f"C1{suffix} (n1{suffix} out{suffix}) capacitor c={C1}")
                lines.append(f"C2{suffix} (n2{suffix} 0) capacitor c={C2}")
                lines.append(f"save v(out{suffix})")

        lines.append("ac dec 50 1 1M")

    elif fmt == "spice":
        lines = [f"* {title}", f"* Lib:{lib} Cell:{cell}", ".options savecurrents", "V1 in 0 AC 1"]
        
        if d_type in ["rc_lp", "rc_hp"]:
            R = eng_suffix(design_data["R"], "R")
            C = eng_suffix(design_data["C"], "C")
            if d_type == "rc_lp":
                lines.append(f"R1 in out {R}")
                lines.append(f"C1 out 0 {C}")
            else:
                lines.append(f"C1 in out {C}")
                lines.append(f"R1 out 0 {R}")
        
        elif d_type == "resonator":
            stages = design_data.get("stages", [])
            for i, st in enumerate(stages):
                idx = i + 1
                s = f"_{idx}"
                R1 = eng_suffix(st["R1"], "R")
                R2 = eng_suffix(st["R2"], "R")
                C1 = eng_suffix(st["C1"], "C")
                C2 = eng_suffix(st["C2"], "C")
                f0 = float(st["f0"])
                Q = float(st["Q"])

                lines.append(f"* Stage {idx} f0={f0:.1f}Hz Q={Q:.2f}")
                lines.append(f"E{idx} out{s} 0 n2{s} 0 1.0")
                lines.append(f"R1{s} in n1{s} {R1}")
                lines.append(f"R2{s} n1{s} n2{s} {R2}")
                lines.append(f"C1{s} n1{s} out{s} {C1}")
                lines.append(f"C2{s} n2{s} 0 {C2}")

        lines.append(".ac dec 50 1 1meg")
        lines.append(".end")

    return "\n".join(lines)

# =========================================================
# Routes
# =========================================================
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request): return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(username: str=Form(...), password: str=Form(...)):
    if authenticate_user(username, password):
        r = RedirectResponse("/dashboard", 303)
        r.set_cookie("logged_in", "true", httponly=True)
        r.set_cookie("username", username, httponly=True)
        return r
    return RedirectResponse("/", 303)

@app.get("/logout")
def logout_route():
    r = RedirectResponse("/", 303)
    r.delete_cookie("logged_in")
    r.delete_cookie("username")
    return r

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request): return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
def register(username: str=Form(...), password: str=Form(...), email: str=Form(...), phone: str=Form(None)):
    users = load_users()
    if any(u["username"] == username for u in users): return RedirectResponse("/register?error=exists", 303)
    users.append({"id": len(users)+1, "username": username, "password": bcrypt.hash(password), "email": email, "phone": phone})
    save_users(users)
    return RedirectResponse("/?registered=1", 303)

@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})

@app.post("/forgot-password")
def forgot_password_submit(email: str=Form(...), new_password: str=Form(...)):
    users = load_users()
    for u in users:
        if u.get("email") == email:
            u["password"] = bcrypt.hash(new_password)
            save_users(users)
            return RedirectResponse("/?message=PasswordResetSuccess", 303)
    return RedirectResponse("/forgot-password?error=EmailNotFound", 303)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if request.cookies.get("logged_in") != "true": return RedirectResponse("/", 303)
    
    username = request.cookies.get("username")
    users = load_users()
    current_user = next((u for u in users if u.get("username") == username), {})
    
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})

@app.post("/update_profile")
def update_profile(request: Request, email: str=Form(...), phone: str=Form(None), new_password: str=Form(None)):
    username = request.cookies.get("username")
    if not username: return RedirectResponse("/", 303)
    
    users = load_users()
    for u in users:
        if u["username"] == username:
            u["email"] = email
            u["phone"] = phone
            if new_password and new_password.strip():
                u["password"] = bcrypt.hash(new_password)
            save_users(users)
            break
    return RedirectResponse("/dashboard?message=ProfileUpdated", 303)

# --- Full Training Stream ---
@app.get("/train_stream")
def train_stream():
    def event_generator():
        try:
            start_time = time.time()
            data_dir = os.path.join("dataset", "train")
            if not os.path.exists(data_dir):
                yield "data: ERROR: dataset/train not found.\n\n"
                return

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            
            train_dataset = datasets.ImageFolder(data_dir, transform=transform)
            if not train_dataset.classes:
                yield "data: ERROR: No classes found.\n\n"
                return

            ml_state.class_names = train_dataset.classes
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

            ml_state.model = SimpleCNN(num_classes=len(ml_state.class_names)).to(ml_state.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(ml_state.model.parameters(), lr=0.001)
            num_epochs = 30

            yield f"data: Classes: {', '.join(ml_state.class_names)}\n\n"

            for epoch in range(num_epochs):
                ml_state.model.train()
                running_loss, correct, total = 0.0, 0, 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(ml_state.device), labels.to(ml_state.device)
                    optimizer.zero_grad()
                    outputs = ml_state.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                avg_loss = running_loss / max(1, len(train_loader))
                acc = 100.0 * correct / max(1, total)
                progress = int(((epoch + 1) / num_epochs) * 100)
                msg = f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, Progress: {progress}%"
                yield f"data: {msg}\n\n"

            torch.save(ml_state.model.state_dict(), "best_model.pth")
            yield "data: Training completed. Model saved.\n\n"

        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
            print(traceback.format_exc())

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- Resume Training Stream ---
@app.get("/train_stream_resume")
def train_stream_resume():
    def event_generator():
        try:
            if ml_state.model is None:
                yield "data: ERROR: No model loaded to resume.\n\n"
                return

            data_dir = os.path.join("dataset", "train")
            if not os.path.exists(data_dir):
                yield "data: ERROR: dataset/train not found.\n\n"
                return

            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            train_dataset = datasets.ImageFolder(data_dir, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            
            ml_state.model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(ml_state.model.parameters(), lr=0.0005) 
            num_epochs = 10

            yield f"data: Resuming training for {num_epochs} epochs...\n\n"

            for epoch in range(num_epochs):
                running_loss, correct, total = 0.0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(ml_state.device), labels.to(ml_state.device)
                    optimizer.zero_grad()
                    outputs = ml_state.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                avg_loss = running_loss / max(1, len(train_loader))
                acc = 100.0 * correct / max(1, total)
                progress = int(((epoch + 1) / num_epochs) * 100)
                msg = f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, Progress: {progress}%"
                yield f"data: {msg}\n\n"

            torch.save(ml_state.model.state_dict(), "best_model.pth")
            yield "data: Training completed. Model saved.\n\n"

        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
            print(traceback.format_exc())

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Prediction ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if ml_state.model is None: return {"error": "Model not loaded"}
    
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = t(img).unsqueeze(0).to(ml_state.device)
    
    ml_state.model.eval()
    with torch.no_grad():
        out = ml_state.model(img)
        conf, idx = torch.max(torch.softmax(out, 1), 1)
    
    label = ml_state.class_names[idx.item()]
    colors = {"amplifier": "#f39c12", "rc_lp": "#2ecc71", "rc_hp": "#1abc9c", "resonator": "#9b59b6", "other": "#95a5a6"}
    friendly = {"rc_lp": "RC Lowpass", "rc_hp": "RC Highpass", "resonator": "Resonator"}
    
    return {
        "class": label,
        "friendly": friendly.get(label, label),
        "color": colors.get(label, "#000"),
        "confidence": round(conf.item() * 100, 2)
    }

# --- Design RC ---
@app.post("/design_rc")
async def design_rc(payload: dict = Body(...)):
    rc_type = payload.get("rc_type")
    fc = float(payload.get("fc_hz", 0))
    if fc <= 0: return {"error": "Invalid cutoff"}
    
    known = payload.get("known", "auto")
    R_in, C_in = payload.get("R_ohm"), payload.get("C_f")
    w_c = 2 * math.pi * fc
    
    if known == "R" and R_in:
        R, C = R_in, 1.0/(w_c*R_in)
    elif known == "C" and C_in:
        C, R = C_in, 1.0/(w_c*C_in)
    else:
        C, R = 100e-9, 1.0/(w_c*100e-9)

    # Bode
    freqs = np.logspace(np.log10(fc/100), np.log10(fc*100), 200)
    if rc_type == "rc_lp":
        def H(s): return 1 / (1 + s*R*C)
    else:
        def H(s): return (s*R*C) / (1 + s*R*C)
        
    mags = calculate_bode(freqs, H)
    
    return {
        "design": {"type": rc_type, "R": R, "C": C},
        "display": {"R": f"{eng_suffix(R, 'R')}Ω", "C": f"{eng_suffix(C, 'C')}F"},
        "bode": {"freqs": freqs.tolist(), "mags": mags}
    }

# --- Vowel Data Handling ---
@app.get("/get_vowels_list")
def get_vowels_list():
    if not VOWEL_DATA_CACHE:
        load_vowel_data() # Try to reload if empty
    return {"vowels": list(VOWEL_DATA_CACHE.keys())}

@app.get("/get_vowel_design/{vowel_name}")
def get_vowel_design(vowel_name: str):
    if vowel_name not in VOWEL_DATA_CACHE:
        return {"error": "Vowel not found"}
    
    stages = VOWEL_DATA_CACHE[vowel_name]
    
    # Calculate Bode for the selected vowel
    f_min, f_max = 100, 5000
    bode_f = np.logspace(np.log10(f_min), np.log10(f_max), 400)
    s_arr = 1j * 2 * np.pi * bode_f
    complex_resp = np.ones_like(bode_f, dtype=complex)
    
    display_comps = []
    
    for i, st in enumerate(stages):
        f0 = float(st["f0"])
        w0 = f0 * 2 * math.pi
        Q = float(st["Q"])
        
        # 2nd Order Lowpass Resonator Transfer Function
        denom = s_arr**2 + s_arr*(w0/Q) + w0**2
        H_stage = (w0**2) / denom
        complex_resp = complex_resp * H_stage
        
        display_comps.append({
            "id": i+1,
            "f0": f"{f0:.1f}",
            "Q": f"{Q:.2f}",
            "R1": eng_suffix(st["R1"], "R")+"Ω",
            "R2": eng_suffix(st["R2"], "R")+"Ω",
            "C1": eng_suffix(st["C1"], "C")+"F",
            "C2": eng_suffix(st["C2"], "C")+"F",
            "peak": st.get("fk", "?"), "bw": st.get("bw", "?")
        })
        
    mag = np.abs(complex_resp)
    mags_db = 20 * np.log10(mag + 1e-15)
    
    return {
        "design": {"type": "resonator", "stages": stages},
        "components": display_comps,
        "bode": {"freqs": bode_f.tolist(), "mags": mags_db.tolist()}
    }

# --- Export Netlist ---
@app.post("/export_netlist")
async def export_netlist(payload: dict = Body(...)):
    fmt = payload.get("format", "spectre")
    data = payload.get("design_data", {})
    lib = payload.get("lib", "")
    cell = payload.get("cell", "")
    view = payload.get("view", "schematic")
    
    text = make_netlist(fmt, data, lib, cell, view)
    fname = f"netlist.{'scs' if fmt=='spectre' else 'cir'}"
    
    return Response(text, media_type="text/plain", headers={"Content-Disposition": f'attachment; filename="{fname}"'})

@app.post("/load_model_default")
def load_def():
    path = "best_model.pth"
    if not os.path.exists(path): return {"status": "error", "message": "best_model.pth not found"}
    state = torch.load(path, map_location=ml_state.device)
    ml_state.model.load_state_dict(state)
    return {"status": "ok", "classes": ml_state.class_names}

@app.post("/load_model_file")
async def load_f(file: UploadFile = File(...)):
    path = os.path.join(MODELS_DIR, file.filename)
    with open(path, "wb") as f: f.write(await file.read())
    try:
        state = torch.load(path, map_location=ml_state.device)
        ml_state.model.load_state_dict(state)
        return {"status": "ok", "classes": ml_state.class_names}
    except:
        return {"status": "error", "message": "Invalid model file"}