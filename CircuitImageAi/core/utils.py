import os
import json
import math
import urllib.parse
import numpy as np

USERS_FILE = "users.json"
VOWELS_FILE = "vowel_results.txt"
E24_VALUES = [
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1
]

# Create a JSON file with default content if it does not already exist.
def ensure_file_exists(filepath: str, default_content):
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_content, f)

# Format a component value using an appropriate engineering suffix.
def eng_suffix(value: float, kind: str) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
        
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

# Return the nearest standard E24 value to the supplied positive value.
def get_nearest_e_value(value: float) -> float:
    if value <= 0: return 0.0
    exponent = math.floor(math.log10(value))
    magnitude = 10 ** exponent
    normalized = value / magnitude
    closest_base = min(E24_VALUES, key=lambda x: abs(x - normalized))
    return round(closest_base * magnitude, 9)

# Generate supplier search links for the specified component value and type.
def generate_supplier_links(value: float, kind: str) -> dict:
    formatted = eng_suffix(value, kind)
    term = f"{formatted}Ohm resistor" if kind == "R" else f"{formatted}F capacitor"
    query = urllib.parse.quote(term)
    return {
        "Digi-Key": f"https://www.digikey.com/en/products/result?keywords={query}",
        "Mouser": f"https://www.mouser.com/c/?q={query}",
        "LCSC": f"https://www.lcsc.com/search?q={query}"
    }

# Calculate the transfer function's magnitude response in decibels.
def calculate_bode(frequencies, H_s_func):
    s = 1j * 2 * np.pi * frequencies
    H = H_s_func(s)
    mag = np.abs(H)
    mag = np.where(mag < 1e-15, 1e-15, mag)
    return (20 * np.log10(mag)).tolist()

# Format polynomial coefficients as a readable expression in S.
def format_poly(coeffs):
    terms = []
    degree = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-10: continue
        p = degree - i
        term = f"{c:.3g}"
        if p > 0: term += " S"
        if p > 1: term += f"^{p}"
        terms.append(term)
    return " + ".join(terms).replace("+ -", "- ")