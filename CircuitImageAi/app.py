# app.py
from fastapi import FastAPI, Request, Form, UploadFile, File, Body, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os, time, math, datetime, traceback
from io import BytesIO
from typing import Generator, Optional

# --- Torch / Vision ---
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

# --- SQLAlchemy + MySQL + bcrypt ---
from sqlalchemy import create_engine, Column, Integer, String, select, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import IntegrityError, OperationalError
from passlib.hash import bcrypt

from urllib.parse import urlparse
from starlette import status

# =========================================================
# FastAPI setup & static/templates
# =========================================================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# =========================================================
# Database (MySQL via SQLAlchemy)
# =========================================================
DATABASE_URL = "mysql+pymysql://circuit_user:StrongPassword123!@127.0.0.1:3306/circuitai"

def ensure_database_exists(url: str):
    """
    If the target database doesn't exist, connect to the server without the DB
    and create it. Works with or without a password.
    """
    parsed = urlparse(url.replace("mysql+pymysql://", "mysql://", 1))
    dbname = (parsed.path or "").lstrip("/")
    if not dbname:
        return  # no db name in URL; nothing to do

    user = parsed.username or ""
    pwd  = parsed.password or ""   # empty is OK
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3306

    # Build auth segment correctly whether password is empty or not
    auth = user if pwd == "" else f"{user}:{pwd}"
    server_url = f"mysql+pymysql://{auth}@{host}:{port}/"

    server_engine = create_engine(server_url, pool_pre_ping=True, future=True)
    with server_engine.connect() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{dbname}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        ))
        conn.commit()

try:
    ensure_database_exists(DATABASE_URL)
except OperationalError:
    # If user lacks CREATE DATABASE, make sure DB exists manually
    pass

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)  # bcrypt hash
    # Keep email present in the ORM. We'll ensure it exists in the DB below.
    # Using nullable=True allows legacy rows without email; registration still requires one.
    email    = Column(String(255), unique=True, nullable=True, index=True)
    phone    = Column(String(20), nullable=True)

def ensure_users_table_and_email_column():
    """
    Create 'users' table if missing. If it exists but is missing 'email',
    add it and a UNIQUE constraint/index. Safe to run multiple times.
    """
    # 1) Ensure base tables exist (create if missing). NOTE: this will NOT alter existing tables.
    Base.metadata.create_all(bind=engine)

    # 2) Check whether 'email' exists in the actual MySQL table; add if missing.
    with engine.begin() as conn:
        # Which schema are we connected to?
        dbname_row = conn.execute(text("SELECT DATABASE()")).first()
        current_db = (dbname_row[0] if dbname_row else None) or ""

        # Does the column exist?
        col_count = conn.execute(
            text("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = :db AND TABLE_NAME = 'users' AND COLUMN_NAME = 'email'
            """),
            {"db": current_db}
        ).scalar()

        if not col_count:
            # Add column as NULLable first to avoid failing on existing rows
            conn.execute(text("ALTER TABLE `users` ADD COLUMN `email` VARCHAR(255) NULL"))
            # Attempt to add UNIQUE constraint; on older MySQL this creates an index.
            # If it already exists or fails, ignore gracefully.
            try:
                conn.execute(text("ALTER TABLE `users` ADD CONSTRAINT `uq_users_email` UNIQUE (`email`)"))
            except Exception:
                pass
            # Add a plain index for faster lookups (if not already created by ORM)
            try:
                conn.execute(text("CREATE INDEX `ix_users_email` ON `users` (`email`)"))
            except Exception:
                pass

# Ensure schema alignment at import-time
try:
    ensure_users_table_and_email_column()
except Exception as e:
    # Don't crash app startup if permissions are limited; you'll just need to run the ALTER manually.
    print(f"[WARN] Could not auto-ensure users.email column: {e}")

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Password helpers ---
def hash_password(plain: str) -> str:
    return bcrypt.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.verify(plain, hashed)
    except Exception:
        return False

def create_user(db: Session, username: str, password: str, email: Optional[str] = None, phone: Optional[str] = None):
    hashed = hash_password(password)
    user = User(username=username, password=hashed, email=email, phone=phone)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db: Session, username: str, password: str) -> bool:
    stmt = select(User).where(User.username == username)
    user = db.execute(stmt).scalars().first()
    if not user:
        return False
    return verify_password(password, user.password)

# =========================================================
# Model & helpers
# =========================================================
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

model = None
class_names = []

label_map = {
    "amplifier": {"friendly": "Amplifier",   "color": "#f39c12"},
    "rc_lp":     {"friendly": "RC Lowpass",  "color": "#2ecc71"},
    "rc_hp":     {"friendly": "RC Highpass", "color": "#1abc9c"},
    "other":     {"friendly": "Other",       "color": "#95a5a6"},
}

def get_class_names():
    data_dir = "dataset"
    train_dir = os.path.join(data_dir, "train")
    ds = datasets.ImageFolder(train_dir)
    return ds.classes

# Pretty units for RC
def format_ohms(r):
    if r >= 1e6:  return f"{r/1e6:.3g} MΩ"
    if r >= 1e3:  return f"{r/1e3:.3g} kΩ"
    return f"{r:.3g} Ω"
def format_farads(c):
    if c >= 1e-3:   return f"{c*1e3:.3g} mF"
    if c >= 1e-6:   return f"{c*1e6:.3g} µF"
    if c >= 1e-9:   return f"{c*1e9:.3g} nF"
    if c >= 1e-12:  return f"{c*1e12:.3g} pF"
    return f"{c:.3g} F"

# =========================================================
# Pages (Auth + Dashboard)
# =========================================================
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    if authenticate_user(db, username, password):
        resp = RedirectResponse("/dashboard", status_code=303)
        resp.set_cookie("logged_in", "true", httponly=True)
        return resp
    return RedirectResponse("/", status_code=303)

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    phone: str = Form(None),
    db: Session = Depends(get_db),
    request: Request = None,
):
    # check duplicates
    existing = db.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first()
    if existing:
        # re-render the register page with an error message
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Username or email already taken"},
            status_code=400,
        )

    # write to DB
    create_user(db, username=username, password=password, email=email, phone=phone)

    # ✅ redirect to the login page (your "/" route)
    return RedirectResponse(url="/?registered=1", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if request.cookies.get("logged_in") != "true":
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("dashboard.html", {"request": request})

# =========================================================
# Training (SSE) with loss, accuracy, elapsed time
# =========================================================
@app.get("/train_stream")
def train_stream():
    def event_generator():
        global model, class_names, device
        try:
            start_time = time.time()

            data_dir = "dataset"
            train_dir = os.path.join(data_dir, "train")
            if not os.path.isdir(train_dir):
                yield "data: ERROR: dataset/train not found.\n\n"
                return

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            train_dataset = datasets.ImageFolder(train_dir, transform=transform)
            if not train_dataset.classes:
                yield "data: ERROR: No class subfolders in dataset/train (e.g., amplifier, rc_lp, rc_hp, other).\n\n"
                return
            if len(train_dataset) == 0:
                yield "data: ERROR: No images found in dataset/train/*.\n\n"
                return

            class_names = train_dataset.classes
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

            model = SimpleCNN(num_classes=len(class_names)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            num_epochs = 30

            yield f"data: Classes: {', '.join(class_names)}\n\n"

            for epoch in range(num_epochs):
                model.train()
                running_loss, correct, total = 0.0, 0, 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
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
                msg = f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%, Progress: {progress}%"
                yield f"data: {msg}\n\n"
                yield ": keep-alive\n\n"

            # save model(s)
            torch.save(model.state_dict(), "best_model.pth")
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"best_{stamp}.pth"))

            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            yield f"data: Training completed in {mins}m {secs}s\n\n"

        except Exception as e:
            yield f"data: ERROR: {e}\n\n"
            yield f"data: {traceback.format_exc(limit=3)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

# =========================================================
# Prediction (with confidence)
# =========================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, class_names
    if model is None:
        return {"error": "Model not trained"}

    image = Image.open(BytesIO(await file.read())).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = class_names[pred_idx.item()]
    friendly = label_map.get(label, {"friendly": label, "color": "#000000"})
    confidence_pct = float(conf.item() * 100.0)

    return {
        "class": label,
        "friendly": friendly["friendly"],
        "color": friendly["color"],
        "confidence": round(confidence_pct, 2)
    }

# =========================================================
# RC Designer (first-order LP/HP)
# =========================================================
@app.post("/design_rc")
async def design_rc(payload: dict = Body(...)):
    rc_type = payload.get("rc_type")
    fc = payload.get("fc_hz")
    known = (payload.get("known") or "auto").lower()
    R_in = payload.get("R_ohm")
    C_in = payload.get("C_f")
    Q = payload.get("Q", None)

    if rc_type not in ("rc_lp", "rc_hp"):
        return {"error": "Design mode is only available for RC Lowpass/Highpass."}
    if not fc or fc <= 0:
        return {"error": "Please provide a positive cutoff frequency (Hz)."}

    two_pi_fc = 2 * math.pi * fc
    q_note = None
    if Q not in (None, "", 0):
        q_note = "Note: Q is ignored for first-order RC filters."

    if known == "c":
        if not C_in or C_in <= 0:
            return {"error": "Please provide C (in Farads) when known='C'."}
        R = 1.0 / (two_pi_fc * C_in)
        C = C_in
    elif known == "r":
        if not R_in or R_in <= 0:
            return {"error": "Please provide R (in Ohms) when known='R'."}
        C = 1.0 / (two_pi_fc * R_in)
        R = R_in
    else:
        C = 100e-9  # 100 nF default
        R = 1.0 / (two_pi_fc * C)

    return {
        "rc_type": rc_type,
        "fc_hz": fc,
        "R_ohm": R,
        "C_f": C,
        "R_pretty": format_ohms(R),
        "C_pretty": format_farads(C),
        "note": q_note
    }

# =========================================================
# Netlist export (Spectre)
# =========================================================
def eng_suffix(value: float, kind: str) -> str:
    if kind == "R":
        if value >= 1e9: return f"{value/1e9:.3g}G"
        if value >= 1e6: return f"{value/1e6:.3g}M"
        if value >= 1e3: return f"{value/1e3:.3g}K"
        return f"{value:.6g}"
    if kind == "C":
        if value < 1e-9:
            if value < 1e-12: return f"{value/1e-15:.3g}f"
            return f"{value/1e-12:.3g}p"
        if value < 1e-6:  return f"{value/1e-9:.3g}n"
        if value < 1e-3:  return f"{value/1e-6:.3g}u"
        if value < 1:     return f"{value/1e-3:.3g}m"
        return f"{value:.6g}"
    return f"{value:.6g}"

def make_rc_spectre(
    rc_type: str,
    R_ohm: float,
    C_f: float,
    title: str = None,
    ac_start: float = 1.0,
    ac_stop: float = 1e6,
    ac_pts: int = 200,
    lib: str = "thesis_prj",
    cell: str = "test",
    view: str = "schematic",
    v_dc: float = 0.0,
    v_acmag: float = 1.0,
    v_type: str = "sine",
    v_ampl: float = 0.01,
    vin_node: str = "in",
    vout_node: str = "ideal_out",
):
    import datetime
    title = title or (f"{'RC Lowpass' if rc_type=='rc_lp' else 'RC Highpass'} "
                      f"auto-generated {datetime.datetime.now():%Y-%m-%d %H:%M}")
    R_str = eng_suffix(float(R_ohm), "R")
    C_str = eng_suffix(float(C_f), "C")

    lines = [
        f"// Library name: {lib}",
        f"// Cell name: {cell}",
        f"// View name: {view}",
        f"// {title}",
        "simulator lang=spectre",
        "global 0",
    ]

    vline = f"V1 ({vin_node} 0) vsource dc={v_dc}"
    if v_type.lower() == "sine":
        vline += f" mag={v_acmag} type=sine ampl={eng_suffix(v_ampl, 'R')}"
    else:
        vline += f" mag={v_acmag}"
    lines.append(vline)

    if rc_type == "rc_lp":
        lines += [
            f"R1 ({vin_node} {vout_node}) resistor r={R_str}",
            f"C1 ({vout_node} 0) capacitor c={C_str}",
        ]
    elif rc_type == "rc_hp":
        lines += [
            f"C1 ({vin_node} {vout_node}) capacitor c={C_str}",
            f"R1 ({vout_node} 0) resistor r={R_str}",
        ]
    else:
        lines.append("// Unsupported type; pass-through")

    lines += [
        f"ac dec {ac_pts} {ac_start} {ac_stop}",
        f"save v({vin_node}) v({vout_node})",
        ""
    ]
    return "\n".join(lines)

@app.post("/export_netlist")
async def export_netlist(payload: dict = Body(...)):
    rc_type = payload.get("rc_type")
    R = payload.get("R_ohm")
    C = payload.get("C_f")
    title = payload.get("title")
    ac_start = float(payload.get("ac_start", 1))
    ac_stop  = float(payload.get("ac_stop", 1e6))
    ac_pts   = int(payload.get("ac_pts", 200))
    fmt = (payload.get("format") or "spice").lower()
    default_name = f"{rc_type}_{datetime.datetime.now():%Y%m%d_%H%M%S}." + ("scs" if fmt == "spectre" else "cir")
    filename = payload.get("filename") or default_name

    if fmt == "spectre":
        lib  = (payload.get("lib") or "").strip()
        cell = (payload.get("cell") or "").strip()
        view = (payload.get("view") or "schematic").strip()
        if not lib or not cell:
            return {"error": "Spectre export requires non-empty 'lib' and 'cell'."}

        text_out = make_rc_spectre(
            rc_type=rc_type,
            R_ohm=float(R),
            C_f=float(C),
            title=title,
            ac_start=ac_start,
            ac_stop=ac_stop,
            ac_pts=ac_pts,
            lib=lib,
            cell=cell,
            view=view,
            # You can expose these as inputs later if you want:
            v_dc=float(payload.get("v_dc", 0.0)),
            v_acmag=float(payload.get("v_acmag", 1.0)),
            v_type=payload.get("v_type", "sine"),
            v_ampl=float(payload.get("v_ampl", 0.01)),
            vin_node=payload.get("vin_node", "in"),
            vout_node=payload.get("vout_node", "ideal_out"),
        )
    else:
        return {"error": "Only Spectre export implemented in this endpoint."}

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
    }
    return Response(content=text_out, media_type="text/plain", headers=headers)

# =========================================================
# Model management (load saved / upload)
# =========================================================
@app.post("/load_model_default")
def load_model_default():
    global model, class_names, device
    weights_path = "best_model.pth"
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="best_model.pth not found. Train first or upload a model.")
    class_names = get_class_names()
    tmp_model = SimpleCNN(num_classes=len(class_names)).to(device)
    state = torch.load(weights_path, map_location=device)
    try:
        tmp_model.load_state_dict(state)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load state dict: {e}")
    model = tmp_model
    return {"status": "ok", "message": "Loaded best_model.pth", "classes": class_names}

@app.post("/load_model_file")
async def load_model_file(file: UploadFile = File(...)):
    global model, class_names, device
    if not file.filename.endswith(".pth"):
        raise HTTPException(status_code=400, detail="Please upload a .pth file.")
    uploads_dir = os.path.join(MODELS_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    save_path = os.path.join(uploads_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    class_names = get_class_names()
    tmp_model = SimpleCNN(num_classes=len(class_names)).to(device)
    try:
        state = torch.load(save_path, map_location=device)
        tmp_model.load_state_dict(state)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load state dict: {e}")
    model = tmp_model
    return {"status": "ok", "message": f"Loaded {file.filename}", "classes": class_names}
