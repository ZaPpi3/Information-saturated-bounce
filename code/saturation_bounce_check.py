import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Ensure figures output directory exists
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_torus_distances(L):
    """Generates translation-invariant distance matrices on a closed torus topology."""
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    pos = np.vstack([x.ravel(), y.ravel()]).T
    N = L * L
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = min(abs(pos[i,0] - pos[j,0]), L - abs(pos[i,0] - pos[j,0]))
                dy = min(abs(pos[i,1] - pos[j,1]), L - abs(pos[i,1] - pos[j,1]))
                D[i,j] = np.sqrt(dx**2 + dy**2)
    return D

def generate_bounce_plots(L=8):
    N = L * L
    D = get_torus_distances(L)
    mask = D > 0
    
    # Smooth, high-density alpha sweep tracking the contraction phase
    alpha_sweep = np.linspace(0.0, 1.5, 100)
    a_scale = 1.05
    
    w_series = []
    gap_series = []
    dim_series = []
    
    for alpha in alpha_sweep:
        # 1. Base State Calculation
        I_base = np.zeros_like(D)
        if alpha == 0:
            I_base[mask] = 1.0
        else:
            I_base[mask] = 1.0 / (D[mask] ** alpha)
        L_base = np.diag(np.sum(I_base, axis=1)) - I_base
        vals_base = np.sort(la.eigvalsh(L_base))
        lambda_1_base = vals_base[1]  # Track true first non-zero eigenvalue
        
        # 2. Deformed State Calculation
        I_scaled = np.zeros_like(D)
        if alpha == 0:
            I_scaled[mask] = 1.0
        else:
            I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha)
        L_scaled = np.diag(np.sum(I_scaled, axis=1)) - I_scaled
        lambda_1_scaled = np.sort(la.eigvalsh(L_scaled))[1]
        
        # 3. Scaling Properties & Parameter Extractions
        log_lambda_ratio = np.log(lambda_1_scaled / lambda_1_base)
        log_a_ratio = np.log(a_scale)
        scaling_exponent = log_lambda_ratio / log_a_ratio
        w_eff = (-scaling_exponent / 3.0) - 1.0
        
        # 4. Weyl Slope Fitting
        max_mode = int(N * 0.25)
        modes = np.arange(1, max_mode)
        slope, _ = np.polyfit(np.log(modes), np.log(vals_base[modes]), 1)
        d_eff = 2.0 / slope if slope > 1e-5 else 100.0  # Cap divergence for plotting stability
        
        w_series.append(w_eff)
        gap_series.append(lambda_1_base)
        dim_series.append(d_eff)

    # =====================================================================
    # PLOT 1: EQUATION OF STATE TRANSLATION (THE NEGATIVE PRESSURE SWITCH)
    # =====================================================================
    plt.figure(figsize=(6.5, 4.5), dpi=300)
    plt.plot(alpha_sweep, w_series, color="purple", linewidth=2.5, label=r"Substrate $w_{\text{eff}}$")
    plt.axhline(-1.0, color="red", linestyle="--", alpha=0.7, label=r"Cosmological Constant ($w = -1$)")
    plt.axvline(0.0, color="black", linestyle=":", alpha=0.5)
    
    plt.title("Information-Saturation Phase Transition: Equation of State", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel(r"Entanglement Decay Exponent $\alpha$ (Localization Index)", fontsize=9.5)
    plt.ylabel(r"Effective Parameter $w_{\text{eff}}$", fontsize=9.5)
    plt.gca().invert_xaxis()  # Invert X axis to represent a spatial contraction/crunch reading right-to-left
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    
    w_fig_path = os.path.join(OUTPUT_DIR, "saturation_bounce_equation_of_state.png")
    plt.savefig(w_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # =====================================================================
    # PLOT 2: SPECTRAL GAP SATURATION & INFRARED CUTOFF
    # =====================================================================
    plt.figure(figsize=(6.5, 4.5), dpi=300)
    plt.plot(alpha_sweep, gap_series, color="navy", linewidth=2.5, label=r"Spectral Gap $\lambda_1$ ($\rho_{\text{eff}}$)")
    plt.axhline(float(N), color="darkgreen", linestyle="-.", alpha=0.7, label=rf"Lattice Capacity Cap ($N = {N}$)")
    
    plt.title("Ultraviolet Energy Density Invariant Saturation Threshold", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel(r"Entanglement Decay Exponent $\alpha$ (Localization Index)", fontsize=9.5)
    plt.ylabel(r"Substrate Energy Metric ($\lambda_1$)", fontsize=9.5)
    plt.gca().invert_xaxis()  # Invert to match crunch timeline
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    
    gap_fig_path = os.path.join(OUTPUT_DIR, "saturation_bounce_energy_cutoff.png")
    plt.savefig(gap_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("--- Visual Output Engine Complete ---")
    print(f"Saved Equation of State tracking curve to: '{w_fig_path}'")
    print(f"Saved Spectral Cutoff threshold profile to: '{gap_fig_path}'")

if __name__ == "__main__":
    generate_bounce_plots(L=8)
