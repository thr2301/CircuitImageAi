import math
import json
from scipy.optimize import fsolve

# --- Configuration from MATLAB ---
VOWEL_NAMES = ['IY', 'IH', 'EH', 'AE', 'AH', 'AA', 'AO', 'ER', 'UH', 'UW', 'OO', 'Ne']

# Formant Frequencies [F1, F2, F3] (from MATLAB F matrix)
F_MATRIX = [
    [270, 2290, 3010],  # 1->IY
    [390, 1990, 2550],  # 2->IH
    [530, 1840, 2480],  # 3->EH
    [660, 1720, 2410],  # 4->AE
    [520, 1190, 2390],  # 5->AH
    [730, 1090, 2390],  # 6->AA
    [570, 840, 2410],   # 7->AO
    [490, 1350, 1690],  # 8->ER
    [440, 1020, 2240],  # 9->UH
    [300, 870, 2240],   # 10->UW
    [300, 870, 2240],   # 11->OO
    [500, 1500, 2500]   # 12->Ne
]

# Bandwidths [BW1, BW2, BW3]
BW_VECTOR = [40, 60, 75]

# Capacitor settings per stage (Fixed)
STAGES_CAPS = [
    {"C1": 7e-9, "C2": 3e-9},   # Stage 1
    {"C1": 10e-9, "C2": 1e-9},  # Stage 2
    {"C1": 3e-9, "C2": 0.5e-9}  # Stage 3
]

OUTPUT_FILE = "vowel_results.txt"

def eng_suffix(value, kind):
    """Helper to format numbers like 10k, 100n"""
    val = float(value)
    if val == 0: return "0"
    if kind == "R":
        if val >= 1e6: return f"{val/1e6:.4g}M"
        if val >= 1e3: return f"{val/1e3:.4g}k"
    if kind == "C":
        if val < 1e-9: return f"{val/1e-12:.4g}p"
        if val < 1e-6: return f"{val/1e-9:.4g}n"
        if val < 1e-3: return f"{val/1e-6:.4g}u"
    return f"{val:.4g}"

def solve_stage_params(fk, bw, C1, C2):
    """
    Solves the nonlinear system for w0 and Q using the MATLAB equations.
    """
    def equations(vars):
        w0, q = vars
        if q < 0.71: q = 0.71 # Avoid complex numbers in sqrt
        
        # Equation 1: Peak Frequency Shift
        # MATLAB: eqn1 =(2*pi*fk)-p1*sqrt(1-1/(2*(q1^2)))== 0;
        term_peak = 1 - 1/(2*q**2)
        if term_peak < 0: term_peak = 0
        eq1 = (2*math.pi*fk) - w0 * math.sqrt(term_peak)
        
        # Equation 2: Bandwidth
        # MATLAB: term1=1-1/(2*(q1^2))+sqrt(2/(q1^2)+1/(2*(q1^4)));
        # MATLAB: term2=1-1/(2*(q1^2))-sqrt(1/(q1^2)+1/(2*(q1^4)));
        # MATLAB: eqn2=(2*pi*BW)-p1*sqrt(term1)+p1*sqrt(term2)==0;
        
        sq_part = math.sqrt(1/(q**2) + 1/(2*q**4)) # Used in both terms (term1 uses 2/q^2 though?)
        # Let's match MATLAB exact text:
        # term1 uses sqrt(2/(q^2)+1/(2*(q^4)))
        sqrt_term1 = math.sqrt(2/(q**2) + 1/(2*q**4))
        term1 = 1 - 1/(2*q**2) + sqrt_term1
        
        # term2 uses sqrt(1/(q^2)+1/(2*(q^4)))
        sqrt_term2 = math.sqrt(1/(q**2) + 1/(2*q**4))
        term2 = 1 - 1/(2*q**2) - sqrt_term2
        
        val1 = math.sqrt(term1) if term1 > 0 else 0
        val2 = math.sqrt(term2) if term2 > 0 else 0
        
        eq2 = (2*math.pi*bw) - w0*val1 + w0*val2
        return [eq1, eq2]

    # Initial guess: w0=2*pi*fk, Q=fk/bw
    guess = [2*math.pi*fk, fk/bw if bw>0 else 10]
    
    try:
        w0, Q = fsolve(equations, guess)
    except:
        w0, Q = 2*math.pi*fk, fk/bw

    f0 = w0 / (2*math.pi)
    
    # Calculate R1, R2
    # MATLAB: BW1 = w01/Q1; eqn1 = BW1 == 1/(R1*C1);
    BW_rad = w0 / Q
    R1 = 1.0 / (BW_rad * C1)
    
    # MATLAB: eqn2 = w01 == 1/sqrt(R1*R2*C1*C2);
    # w0^2 = 1/(R1*R2*C1*C2) -> R2 = 1/(w0^2 * R1 * C1 * C2)
    R2 = 1.0 / (w0**2 * R1 * C1 * C2)
    
    return {
        "f0": f0, "w0": w0, "Q": Q, 
        "R1": R1, "R2": R2, "C1": C1, "C2": C2,
        "fk": fk, "bw": bw
    }

def main():
    print("Calculating circuits for 12 vowels (MATLAB Algorithm)...")
    results = {}
    
    for i, name in enumerate(VOWEL_NAMES):
        print(f"Processing {name}...")
        freqs = F_MATRIX[i]
        bws = BW_VECTOR
        
        stages_data = []
        for stage_idx in range(3):
            fk = freqs[stage_idx]
            bw = bws[stage_idx]
            caps = STAGES_CAPS[stage_idx]
            
            res = solve_stage_params(fk, bw, caps["C1"], caps["C2"])
            stages_data.append(res)
            
        results[name] = stages_data
        
    # Save as JSON for easy reading by app.py
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSuccess! Results saved to {OUTPUT_FILE}")
    print("Run app.py now.")

if __name__ == "__main__":
    main()