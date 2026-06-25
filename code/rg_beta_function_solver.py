import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Output architecture
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_erdos_renyi_base(N, p=0.4):
    rng = np.random.default_rng(seed=42)
    adj = rng.random((N, N)) < p
    adj = np.triu(adj, 1) + np.triu(adj, 1).T
    D = np.where(adj, 1.0, 2.5)
    np.fill_diagonal(D, 0.0)
    return D

def generate_watts_strogatz_base(N, k=4, p=0.2):
    rng = np.random.default_rng(seed=42)
    D = np.zeros((N, N))
    for i in range(N):
        for step in range(1, k // 2 + 1):
            D[i, (i + step) % N] = 1.0
            D[i, (i - step) % N] = 1.0
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] == 1.0 and rng.random() < p:
                D[i, j] = 2.5
                new_target = rng.integers(0, N)
                if new_target != i:
                    D[i, new_target] = 1.0
    D = np.maximum(D, D.T)
    D[D == 0.0] = 3.0
    np.fill_diagonal(D, 0.0)
    return D

def generate_barabasi_albert_base(N, m=3):
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

def extract_rg_beta_functions(N=100, steps=80):
    alpha_space = np.linspace(1.5, 0.005, steps)
    topologies = ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]
    a_scale = 1.05
    
    raw_data = {top: {"ln_lambda": [], "w_eff": [], "d_s": []} for top in topologies}
    
    print("--- Extracting Numerical Renormalization Group Beta-Functions ---")
    
    matrix_factory = {
        "Erdos-Renyi": generate_erdos_renyi_base(N),
        "Watts-Strogatz": generate_watts_strogatz_base(N),
        "Barabasi-Albert": generate_barabasi_albert_base(N, m=3)
    }
    
    # 1. Gather raw continuous trajectory metrics
    for alpha in alpha_space:
        for top in topologies:
            D = matrix_factory[top]
            mask = D > 0
            
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
            
            raw_data[top]["ln_lambda"].append(np.log(lambda_1_base))
            raw_data[top]["w_eff"].append(w_eff)
            raw_data[top]["d_s"].append(d_s_peak)

    # 2. Compute Derivatives via Centered Finite Differences
    beta_results = {top: {"ln_lambda_mid": [], "beta_w": [], "beta_ds": []} for top in topologies}
    
    for top in topologies:
        ln_lam = np.array(raw_data[top]["ln_lambda"])
        w_vals = np.array(raw_data[top]["w_eff"])
        ds_vals = np.array(raw_data[top]["d_s"])
        
        # Center stencil loop over interior points
        for idx in range(1, len(ln_lam) - 1):
            d_ln_lam = ln_lam[idx+1] - ln_lam[idx-1]
            
            # Avoid unexpected local coordinate singularity divides
            if abs(d_ln_lam) > 1e-8:
                beta_w = (w_vals[idx+1] - w_vals[idx-1]) / d_ln_lam
                beta_ds = (ds_vals[idx+1] - ds_vals[idx-1]) / d_ln_lam
                
                beta_results[top]["ln_lambda_mid"].append(ln_lam[idx])
                beta_results[top]["beta_w"].append(beta_w)
                beta_results[top]["beta_ds"].append(beta_ds)

    # =====================================================================
    # VISUAL COMPILATION OF THE BETA-FUNCTION TRAJECTORIES
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    colors = {"Erdos-Renyi": "crimson", "Watts-Strogatz": "royalblue", "Barabasi-Albert": "darkmagenta"}
    markers = {"Erdos-Renyi": "o", "Watts-Strogatz": "s", "Barabasi-Albert": "^"}
    
    print(f"\n{'Topology':<16}{'Final Energy ln(λ1)':<22}{'Final β_w':<14}{'Final β_d_s'}")
    print("-" * 68)
    
    for top in topologies:
        x_energy = beta_results[top]["ln_lambda_mid"]
        
        # Left Panel: Equation of State Beta Function beta_w
        ax1.plot(x_energy, beta_results[top]["beta_w"], color=colors[top], linewidth=2, label=f"{top}")
        ax1.plot(x_energy[-1], beta_results[top]["beta_w"][-1], marker=markers[top], color=colors[top], markersize=6)
        
        # Right Panel: Spectral Dimension Beta Function beta_ds
        ax2.plot(x_energy, beta_results[top]["beta_ds"], color=colors[top], linewidth=2, label=f"{top}")
        ax2.plot(x_energy[-1], beta_results[top]["beta_ds"][-1], marker=markers[top], color=colors[top], markersize=6)
        
        print(f"{top:<16}{x_energy[-1]:<22.4f}{beta_results[top]['beta_w'][-1]:<14.4e}{beta_results[top]['beta_ds'][-1]:.4e}")

    # Layout tuning
    ax1.axhline(0.0, color="black", linestyle="--", alpha=0.4)
    ax1.set_title(r"Equation of State Beta-Function $\beta_w$", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Scaling Log-Energy Metric $\ln \lambda_1$", fontsize=9.5)
    ax1.set_ylabel(r"$\beta_w = dw_{\text{eff}} / d\ln\lambda_1$", fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8.5)

    ax2.axhline(0.0, color="black", linestyle="--", alpha=0.4)
    ax2.set_title(r"Spectral Dimension Beta-Function $\beta_{d_s}$", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Scaling Log-Energy Metric $\ln \lambda_1$", fontsize=9.5)
    ax2.set_ylabel(r"$\beta_{d_s} = dd_s / d\ln\lambda_1$", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "cosmological_rg_beta_functions.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print("-" * 68)
    print(f"[Historic Extraction Confirmed]: Saved beta profiles directly to '{fig_path}'")

if __name__ == "__main__":
    extract_rg_beta_functions(N=100, steps=80)
