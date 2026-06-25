import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

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

def execute_rg_flow_simulation(N=100, steps=80):
    alpha_space = np.linspace(1.5, 0.0, steps)
    topologies = ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]
    a_scale = 1.05
    
    results = {top: {"w_eff": [], "lambda_1": []} for top in topologies}
    
    print("--- Simulating Cosmological Renormalization Group (RG) Flow Trajectories ---")
    print(f"{'Topology':<16}{'Alpha (t)':<12}{'w_eff':<12}{'Spectral Gap (λ1)'}")
    print("-" * 56)
    
    matrix_factory = {
        "Erdos-Renyi": generate_erdos_renyi_base(N),
        "Watts-Strogatz": generate_watts_strogatz_base(N),
        "Barabasi-Albert": generate_barabasi_albert_base(N, m=3)
    }
    
    for i, alpha in enumerate(alpha_space):
        for top in topologies:
            D = matrix_factory[top]
            mask = D > 0
            
            # Construct Base Relational Density Matrix
            I_base = np.zeros_like(D)
            if alpha == 0:
                I_base[mask] = 1.0
            else:
                I_base[mask] = 1.0 / (D[mask] ** alpha)
            L_base = np.diag(np.sum(I_base, axis=1)) - I_base
            lambda_1_base = np.sort(la.eigvalsh(L_base))[1]
            
            # Construct Deformed State Matrix
            I_scaled = np.zeros_like(D)
            if alpha == 0:
                I_scaled[mask] = 1.0
            else:
                I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha)
            L_scaled = np.diag(np.sum(I_scaled, axis=1)) - I_scaled
            lambda_1_scaled = np.sort(la.eigvalsh(L_scaled))[1]
            
            # Parameter Extractions
            log_lambda_ratio = np.log(lambda_1_scaled / lambda_1_base)
            log_a_ratio = np.log(a_scale)
            scaling_exponent = log_lambda_ratio / log_a_ratio
            w_eff = (-scaling_exponent / 3.0) - 1.0
            
            results[top]["w_eff"].append(w_eff)
            results[top]["lambda_1"].append(lambda_1_base)
            
        if i % 15 == 0 or i == steps - 1:
            print(f"{'Erdos-Renyi':<16}{alpha:<12.3f}{results['Erdos-Renyi']['w_eff'][-1]:<12.4f}{results['Erdos-Renyi']['lambda_1'][-1]:.4f}")
            print(f"{'Watts-Strogatz':<16}{alpha:<12.3f}{results['Watts-Strogatz']['w_eff'][-1]:<12.4f}{results['Watts-Strogatz']['lambda_1'][-1]:.4f}")
            print(f"{'Barabasi-Albert':<16}{alpha:<12.3f}{results['Barabasi-Albert']['w_eff'][-1]:<12.4f}{results['Barabasi-Albert']['lambda_1'][-1]:.4f}")
            print("-" * 56)

    # =====================================================================
    # GENERATING THE COSMOLOGICAL PHASE PORTRAIT (THE RG FLOW MAP)
    # =====================================================================
    plt.figure(figsize=(7, 5), dpi=300)
    colors = {"Erdos-Renyi": "crimson", "Watts-Strogatz": "royalblue", "Barabasi-Albert": "darkmagenta"}
    markers = {"Erdos-Renyi": "o", "Watts-Strogatz": "s", "Barabasi-Albert": "^"}
    
    for top in topologies:
        # Plotting the explicit vector field trajectory: w_eff vs. Energy Proxy (lambda_1)
        plt.plot(results[top]["lambda_1"], results[top]["w_eff"], color=colors[top], 
                 linewidth=2, alpha=0.8, label=f"{top} Universality Track")
        
        # Add a clear marker on the final saturation endpoint to isolate the fixed point
        plt.plot(results[top]["lambda_1"][-1], results[top]["w_eff"][-1], marker=markers[top], 
                 color=colors[top], markersize=8, zorder=5)
        
    plt.axhline(-1.0, color="red", linestyle="--", alpha=0.6, label="de Sitter Attractor Fixed Point ($w = -1$)")
    
    # Visual cues for directional RG vector flow
    plt.title("Cosmological Renormalization Group (RG) Phase Portrait", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel(r"Substrate Energy Density Scale Proxy ($\lambda_1 \propto \rho_{\text{eff}}$)", fontsize=9.5)
    plt.ylabel(r"Effective Cosmological Fluid Metric $w_{\text{eff}}$", fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, "cosmological_rg_flow_trajectory.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[RG Flow Mapping Complete]: Visual portrait saved directly to '{fig_path}'")

if __name__ == "__main__":
    execute_rg_flow_simulation(N=100, steps=80)
