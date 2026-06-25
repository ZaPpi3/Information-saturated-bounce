import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Output management
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

def generate_3d_rg_manifold(N=100, steps=60):
    alpha_space = np.linspace(1.5, 0.005, steps)  # Safe boundary cutoff
    topologies = ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]
    a_scale = 1.05
    
    results = {top: {"x_lambda": [], "y_weff": [], "z_ds": []} for top in topologies}
    
    print("--- Mapping the 3D Renormalization Group (RG) Cosmological Manifold ---")
    
    matrix_factory = {
        "Erdos-Renyi": generate_erdos_renyi_base(N),
        "Watts-Strogatz": generate_watts_strogatz_base(N),
        "Barabasi-Albert": generate_barabasi_albert_base(N, m=3)
    }
    
    for alpha in alpha_space:
        for top in topologies:
            D = matrix_factory[top]
            mask = D > 0
            
            # Base matrix calculation
            I = np.zeros_like(D)
            I[mask] = 1.0 / (D[mask] ** alpha)
            L_base = np.diag(np.sum(I, axis=1)) - I
            eigenvals = np.sort(la.eigvalsh(L_base))
            lambda_1_base = eigenvals[1]  # FIXED: Isolate scalar spectral gap mode
            
            # Scaled matrix calculation
            I_scaled = np.zeros_like(D)
            I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha)
            L_scaled = np.diag(np.sum(I_scaled, axis=1)) - I_scaled
            lambda_1_scaled = np.sort(la.eigvalsh(L_scaled))[1]  # FIXED: Isolate scalar scaled mode
            
            # Parameter tracking
            scaling_exponent = np.log(lambda_1_scaled / lambda_1_base) / np.log(a_scale)
            w_eff = (-scaling_exponent / 3.0) - 1.0
            d_s_peak = compute_peak_spectral_dimension(eigenvals)
            
            # Append variables into 1D coordinate streams
            results[top]["x_lambda"].append(lambda_1_base)
            results[top]["y_weff"].append(w_eff)
            results[top]["z_ds"].append(d_s_peak)

    # =====================================================================
    # 3D PHASE MANIFOLD COMPILATION
    # =====================================================================
    fig = plt.figure(figsize=(9, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {"Erdos-Renyi": "crimson", "Watts-Strogatz": "royalblue", "Barabasi-Albert": "darkmagenta"}
    markers = {"Erdos-Renyi": "o", "Watts-Strogatz": "s", "Barabasi-Albert": "^"}
    
    for top in topologies:
        # Plot continuous scalar trajectories in 3D grid space
        ax.plot(results[top]["x_lambda"], results[top]["y_weff"], results[top]["z_ds"],
                color=colors[top], linewidth=2.5, alpha=0.8, label=f"{top} Manifold Track")
        
        # Lock final convergence endpoint marker
        ax.plot([results[top]["x_lambda"][-1]], [results[top]["y_weff"][-1]], [results[top]["z_ds"][-1]],
                marker=markers[top], color=colors[top], markersize=8, zorder=10)
        
    ax.set_title("The Universal 3D Renormalization Group Manifold Flow", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel(r"Spectral Gap Scale $\lambda_1$ ($\rho_{\text{eff}}$)", fontsize=9.5, labelpad=8)
    ax.set_ylabel(r"Equation of State $w_{\text{eff}}$", fontsize=9.5, labelpad=8)
    ax.set_zlabel(r"Spectral Dimension $d_s$", fontsize=9.5, labelpad=8)
    
    # Position camera to display optimal multivariable fusion trajectories
    ax.view_init(elev=22, azim=-50)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, "cosmological_3d_rg_manifold.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Manifold Locked Error-Free]: 3D Phase portrait saved to '{fig_path}'")

if __name__ == "__main__":
    generate_3d_rg_manifold(N=100, steps=60)
