import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_barabasi_albert_base(N=100, m=3):
    rng = np.random.default_rng(seed=42)
    D = np.full((N, N), 3.0)
    np.fill_diagonal(D, 0.0)
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = 1.0
    for i in range(m, N):
        degrees = np.sum(D[:i, :i] == 1.0, axis=1)
        prob = degrees / np.sum(degrees)
        targets = rng.choice(np.arange(i), size=m, replace=False, p=prob)
        for t in targets:
            D[i, t] = D[t, i] = 1.0
    return D

def compute_peak_spectral_dimension(eigenvals, t_space=np.logspace(-2, 1, 100)):
    d_s_max = 0.0
    for t in t_space:
        exp_terms = np.exp(-t * eigenvals)
        P_t = np.sum(exp_terms)
        P_prime_t = np.sum(eigenvals * exp_terms)
        d_s_t = 2.0 * t * (P_prime_t / P_t)
        if d_s_t > d_s_max:
            d_s_max = d_s_t
    return d_s_max

def run_linearized_stability_analysis(N=100, steps=120):
    alpha_space = np.linspace(1.5, 0.005, steps)
    a_scale = 1.05
    
    ln_lambda_series = []
    w_series = []
    ds_series = []
    
    D = generate_barabasi_albert_base(N)
    mask = D > 0
    
    print("--- Simulating Universal Baseline Phase Space Trajectory ---")
    for alpha in alpha_space:
        I = np.zeros_like(D)
        I[mask] = 1.0 / (D[mask] ** alpha)
        L_base = np.diag(np.sum(I, axis=1)) - I
        eigenvals = np.sort(la.eigvalsh(L_base))
        lambda_1_base = eigenvals[1]
        
        I_scaled = np.zeros_like(D)
        I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha)
        L_scaled = np.diag(np.sum(I_scaled, axis=1)) - I_scaled
        lambda_1_scaled = np.sort(la.eigvalsh(L_scaled))[1]
        
        scaling_exponent = np.log(lambda_1_scaled / lambda_1_base) / np.log(a_scale)
        w_eff = (-scaling_exponent / 3.0) - 1.0
        d_s_peak = compute_peak_spectral_dimension(eigenvals)
        
        ln_lambda_series.append(np.log(lambda_1_base))
        w_series.append(w_eff)
        ds_series.append(d_s_peak)
        
    ln_lam = np.array(ln_lambda_series)
    w_vals = np.array(w_series)
    ds_vals = np.array(ds_series)
    
    # Compute Raw Derivatives
    beta_w_raw = np.zeros(len(ln_lam)-2)
    beta_ds_raw = np.zeros(len(ln_lam)-2)
    mid_ln_lam = ln_lam[1:-1]
    
    for idx in range(1, len(ln_lam) - 1):
        d_ln_lam = ln_lam[idx+1] - ln_lam[idx-1]
        beta_w_raw[idx-1] = (w_vals[idx+1] - w_vals[idx-1]) / d_ln_lam
        beta_ds_raw[idx-1] = (ds_vals[idx+1] - ds_vals[idx-1]) / d_ln_lam

    # Apply Savitzky-Golay Filter for Analytical Smoothing
    beta_w_smooth = savgol_filter(beta_w_raw, window_length=15, polyorder=2)
    beta_ds_smooth = savgol_filter(beta_ds_raw, window_length=15, polyorder=2)
    
    # Isolate Critical Neighborhood Near Fixed Point (Last 15% of the flow)
    fit_window = int(len(mid_ln_lam) * 0.15)
    fit_idx = slice(-fit_window, -1)
    
    # Linear Regression to Extract Exponents
    # beta_w = -A * (w + 1)
    A_slope, _ = np.polyfit(w_vals[1:-1][fit_idx] + 1.0, beta_w_smooth[fit_idx], 1)
    critical_exponent_A = -A_slope
    
    # beta_ds = -B * (d_s - d_s*)
    ds_star = ds_vals[-1]
    B_slope, _ = np.polyfit(ds_vals[1:-1][fit_idx] - ds_star, beta_ds_smooth[fit_idx], 1)
    critical_exponent_B = -B_slope
    
    print("\n========================================================")
    print("         EXTRACTED LINEARIZED STABILITY EXPONENTS       ")
    print("========================================================")
    print(f"Fluid Flow Critical Exponent (A):      {critical_exponent_A:.4f}")
    print(f"Geometric Scalar Critical Exponent (B): {critical_exponent_B:.4f}")
    print("========================================================")

    # Plot Smoothed Scaling Functions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    ax1.plot(mid_ln_lam, beta_w_smooth, 'darkmagenta', linewidth=2.5, label='Smoothed $\\beta_w$ Flow')
    ax1.axhline(0.0, color='black', linestyle='--', alpha=0.3)
    ax1.set_title(rf"Fluid Stability: $A = {critical_exponent_A:.3f}$", fontsize=11, fontweight='bold')
    ax1.set_xlabel(r"Scaling Log-Energy $\ln \lambda_1$", fontsize=9.5)
    ax1.set_ylabel(r"$\beta_w = dw_{\text{eff}}/d\ln\lambda_1$", fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    
    ax2.plot(mid_ln_lam, beta_ds_smooth, 'royalblue', linewidth=2.5, label='Smoothed $\\beta_{d_s}$ Flow')
    ax2.axhline(0.0, color='black', linestyle='--', alpha=0.3)
    ax2.set_title(rf"Geometric Stability: $B = {critical_exponent_B:.3f}$", fontsize=11, fontweight='bold')
    ax2.set_xlabel(r"Scaling Log-Energy $\ln \lambda_1$", fontsize=9.5)
    ax2.set_ylabel(r"$\beta_{d_s} = dd_s/d\ln\lambda_1$", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "smoothed_rg_stability_exponents.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Analysis Success]: Saved high-impact chart to '{fig_path}'")

if __name__ == "__main__":
    run_linearized_stability_analysis()
