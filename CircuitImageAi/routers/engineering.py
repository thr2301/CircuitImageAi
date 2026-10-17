import io
import math
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import APIRouter, Request, Body, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.utils import VOWELS_FILE, eng_suffix, get_nearest_e_value, generate_supplier_links, calculate_bode, format_poly
from services.netlist_exporter import NetlistFactory
import json
import os

router = APIRouter()
templates = Jinja2Templates(directory="templates")

VOWEL_DATA_CACHE = {}

# Loads vowel filter designs from the configured JSON file into the cache.
def load_vowel_data():
    global VOWEL_DATA_CACHE
    if not os.path.exists(VOWELS_FILE):
        return
    try:
        with open(VOWELS_FILE, "r", encoding="utf-8") as f:
            VOWEL_DATA_CACHE = json.load(f)
    except Exception as e:
        print(f"Error loading {VOWELS_FILE}: {e}")

# Calculates an RC filter design and returns its component values and Bode data.
@router.post("/design_rc")
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

    freqs = np.logspace(np.log10(fc/100), np.log10(fc*100), 200)
    if rc_type == "rc_lp":
        H = lambda s: 1 / (1 + s*R*C)
    else:
        H = lambda s: (s*R*C) / (1 + s*R*C)
        
    mags = calculate_bode(freqs, H)
    
    return {
        "design": {"type": rc_type, "R": R, "C": C},
        "display": {"R": f"{eng_suffix(R, 'R')}Ω", "C": f"{eng_suffix(C, 'C')}F"},
        "bode": {"freqs": freqs.tolist(), "mags": mags}
    }

# Returns the names of the available vowel filter designs.
@router.get("/get_vowels_list")
def get_vowels_list():
    if not VOWEL_DATA_CACHE:
        load_vowel_data()
    return {"vowels": list(VOWEL_DATA_CACHE.keys())}

# Returns a vowel filter design, its component details, and its Bode response.
@router.get("/get_vowel_design/{vowel_name}")
def get_vowel_design(vowel_name: str):
    if not VOWEL_DATA_CACHE:
        load_vowel_data()
    if vowel_name not in VOWEL_DATA_CACHE:
        return {"error": "Vowel not found"}
    
    stages = VOWEL_DATA_CACHE[vowel_name]
    f_min, f_max = 100, 5000
    bode_f = np.logspace(np.log10(f_min), np.log10(f_max), 400)
    s_arr = 1j * 2 * np.pi * bode_f
    complex_resp = np.ones_like(bode_f, dtype=complex)
    display_comps = []
    
    for i, st in enumerate(stages):
        f0 = float(st["f0"])
        w0 = f0 * 2 * math.pi
        Q = float(st["Q"])
        
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

# Exports the supplied circuit design as a downloadable netlist.
@router.post("/export_netlist")
async def export_netlist(payload: dict = Body(...)):
    fmt = payload.get("format", "spectre")
    data = payload.get("design_data", {})
    lib = payload.get("lib", "")
    cell = payload.get("cell", "")
    view = payload.get("view", "schematic")
    
    exporter = NetlistFactory.get_exporter(fmt)
    text = exporter.generate(data, lib, cell, view)
    extensions = {
        "spectre": "scs",
        "spice": "cir",
        "ltspice": "cir",
        "kicad": "net",
    }
    ext = extensions[fmt.strip().lower()]
    fname = f"netlist.{ext}"
    media_type = "application/xml" if fmt.strip().lower() == "kicad" else "text/plain"
    
    return Response(text, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{fname}"'})

class FilterInput(BaseModel):
    frequency: float
    c_base: float
    filter_type: str
    order: int
    q_factor: float

# Serves the filter calculator page to authenticated users.
@router.get("/calculator", response_class=HTMLResponse)
def synthesizer_page(request: Request):
    if request.cookies.get("logged_in") != "true": 
        return RedirectResponse("/", 303)
    return templates.TemplateResponse("calculator.html", {"request": request})

# Synthesizes filter components and returns its transfer function and Bode plot.
@router.post("/calculate")
def calculate_filter(data: FilterInput):
    fc = data.frequency
    C = data.c_base
    n = data.order
    f_type = data.filter_type
    Q = data.q_factor
    
    if fc <= 0 or C <= 0 or Q <= 0:
        raise HTTPException(status_code=400, detail="Values must be > 0.")
    if n < 1:
        raise HTTPException(status_code=400, detail="Order must be at least 1.")
        
    w0 = 2 * np.pi * fc
    components = []

    # Builds a component record with standard values and supplier links.
    def make_comp_dict(stage_name, r_vals=None, c_vals=None, note=None):
        d = {"stage": stage_name}
        if note: d["Note"] = note
        if r_vals:
            for k, v in r_vals.items():
                std_v = get_nearest_e_value(v)
                d[k] = v
                d[f"{k}_std"] = std_v
                d[f"{k}_links"] = generate_supplier_links(std_v, "R")
        if c_vals:
            for k, v in c_vals.items():
                std_v = get_nearest_e_value(v)
                d[k] = v
                d[f"{k}_std"] = std_v
                d[f"{k}_links"] = generate_supplier_links(std_v, "C")
        return d

    if f_type in ["bandpass", "notch"]:
        sections_2nd = n
        sections_1st = 0
        components.append(make_comp_dict(
            "All Stages", 
            note="Unity-Gain Sallen-Key is unstable for Bandpass/Notch. Use Multiple-Feedback (MFB) or State-Variable topologies instead."
        ))
    else:
        sections_2nd = n // 2
        sections_1st = n % 2
        
        for i in range(sections_2nd):
            if f_type == "lowpass":
                m = max(1.0, 4 * (Q**2)) + 0.1
                C2 = C
                C1 = m * C2
                R = 1 / (w0 * np.sqrt(C1 * C2))
                components.append(make_comp_dict(
                    f"Stage {i+1} (2nd Order Low-pass, Q={Q})",
                    r_vals={"R1": R, "R2": R},
                    c_vals={"C1": C1, "C2": C2}
                ))
            elif f_type == "highpass":
                m = max(1.0, 4 * (Q**2)) + 0.1
                C1 = C
                C2 = C
                R1 = 1 / (w0 * C * np.sqrt(m))
                R2 = m * R1
                components.append(make_comp_dict(
                    f"Stage {i+1} (2nd Order High-pass, Q={Q})",
                    r_vals={"R1": R1, "R2": R2},
                    c_vals={"C1": C1, "C2": C2}
                ))
            elif f_type == "allpass":
                components.append(make_comp_dict(
                    f"Stage {i+1} (2nd Order All-pass)",
                    note="Requires complex phase-shifting topology with differential Op-Amp inputs."
                ))

        for i in range(sections_1st):
            R = 1 / (w0 * C)
            components.append(make_comp_dict(
                f"Stage {sections_2nd + i + 1} (1st Order {f_type.capitalize()})",
                r_vals={"R": R},
                c_vals={"C": C}
            ))

    a1, b1_lp, b1_hp, b1_ap = [1, 1], [1], [1, 0], [1, -1]
    a2, b2_lp, b2_hp = [1, 1/Q, 1], [1], [1, 0, 0]
    b2_bp, b2_notch, b2_ap = [1/Q, 0], [1, 0, 1], [1, -1/Q, 1]
    
    b, a = [1], [1]
    for _ in range(sections_2nd):
        a = np.convolve(a, a2)
        if f_type == "lowpass": b = np.convolve(b, b2_lp)
        elif f_type == "highpass": b = np.convolve(b, b2_hp)
        elif f_type == "bandpass": b = np.convolve(b, b2_bp)
        elif f_type == "notch": b = np.convolve(b, b2_notch)
        elif f_type == "allpass": b = np.convolve(b, b2_ap)
        
    for _ in range(sections_1st):
        a = np.convolve(a, a1)
        if f_type == "lowpass": b = np.convolve(b, b1_lp)
        elif f_type == "highpass": b = np.convolve(b, b1_hp)
        elif f_type == "allpass": b = np.convolve(b, b1_ap)

    tf_str = f"H(s) = ({format_poly(b)}) / ({format_poly(a)})"
    
    f = np.logspace(np.log10(fc) - 2, np.log10(fc) + 2, 500)
    s_norm = 1j * (2 * np.pi * f) / w0 
    H_val = np.polyval(b, s_norm) / np.polyval(a, s_norm)
        
    mag_dB = 20 * np.log10(np.abs(H_val))
    phase_deg = np.angle(H_val, deg=True)
    
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.semilogx(f, mag_dB, color='blue')
    plt.axvline(fc, color='red', linestyle='--', label=f'f0 = {fc:.2f} Hz')
    plt.title(f'Bode Plot: Order {n} {f_type.capitalize()} Filter (Q={Q})')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.semilogx(f, phase_deg, color='green')
    plt.axvline(fc, color='red', linestyle='--')
    plt.ylabel('Phase (degrees)')
    plt.xlabel('Frequency (Hz)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return {
        "transfer_function": tf_str,
        "components": components,
        "plot_image_base64": base64.b64encode(buf.read()).decode('utf-8')
    }