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

def compute_peak_spectral_dimension(eigenvals, t_space=np.logspace(-2, 1, 100)):
    """Extracts the peak spectral dimension d_s probing intermediate infrastructure."""
    d_s_max = 0.0
    for t in t_space:
        exp_terms = np.exp(-t * eigenvals)
        P_t = np.sum(exp_terms)
        P_prime_t = np.sum(eigenvals * exp_terms)
        d_s_t = 2.0 * t * (P_prime_t / P_t)
        if d_s_t > d_s_max:
            d_s_max = d_s_t
    return d_s_max

def execute_dimension_rg_flow(N=100, steps=50):
    alpha_space = np.linspace(1.5, 0.001, steps)  # Evade analytical alpha=0 divide
    topologies = ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]
    
    results = {top: {"d_s": [], "lambda_1": []} for top in topologies}
    
    print("--- Mapping Dimensional Renormalization Group (RG) Flow ---")
    print(f"{'Topology':<16}{'Alpha':<10}{'λ1 Proxy':<14}{'Spectral Dim (d_s)'}")
    print("-" * 58)
    
    matrix_factory = {
        "Erdos-Renyi": generate_erdos_renyi_base(N),
        "Watts-Strogatz": generate_watts_strogatz_base(N),
        "Barabasi-Albert": generate_barabasi_albert_base(N, m=3)
    }
    
    for i, alpha in enumerate(alpha_space):
        for top in topologies:
            D = matrix_factory[top]
            mask = D > 0
            
            I = np.zeros_like(D)
            I[mask] = 1.0 / (D[mask] ** alpha)
            L_base = np.diag(np.sum(I, axis=1)) - I
            eigenvals = np.sort(la.eigvalsh(L_base))
            
            d_s_peak = compute_peak_spectral_dimension(eigenvals)
            
            results[top]["d_s"].append(d_s_peak)
            results[top]["lambda_1"].append(eigenvals[1])
            
        if i % 10 == 0 or i == steps - 1:
            print(f"{'Watts-Strogatz':<16}{alpha:<10.3f}{results['Watts-Strogatz']['lambda_1'][-1]:<14.4f}{results['Watts-Strogatz']['d_s'][-1]:.4f}")

    # =====================================================================
    # PHASE PORTRAIT GENERATION: DIMENSIONAL LIQUEFACTION FLOW
    # =====================================================================
    plt.figure(figsize=(7, 5), dpi=300)
    colors = {"Erdos-Renyi": "crimson", "Watts-Strogatz": "royalblue", "Barabasi-Albert": "darkmagenta"}
    markers = {"Erdos-Renyi": "o", "Watts-Strogatz": "s", "Barabasi-Albert": "^"}
    
    for top in topologies:
        plt.plot(results[top]["lambda_1"], results[top]["d_s"], color=colors[top], 
                 linewidth=2, alpha=0.8, label=f"{top} Geometry Track")
        
        # Explicit marker at final simulated step approaching alpha -> 0
        plt.plot(results[top]["lambda_1"][-1], results[top]["d_s"][-1], marker=markers[top], 
                 color=colors[top], markersize=8, zorder=5)
        
    plt.title("Geometric RG Phase Portrait: Spectral Dimension Convergence", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel(r"Substrate Energy Density Scale Proxy ($\lambda_1 \propto \rho_{\text{eff}}$)", fontsize=9.5)
    plt.ylabel(r"Effective Spectral Dimension $d_s$", fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, "spectral_dimension_rg_flow.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Dimensional Mapping Complete]: Saved portrait directly to '{fig_path}'")

if __name__ == "__main__":
    execute_dimension_rg_flow(N=100, steps=50)
