import numpy as np
import scipy.linalg as la

def get_torus_distances(L):
    """Generates a closed, translation-invariant toroidal coordinate grid."""
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

def compute_thermodynamic_energy(eigenvals, beta=0.1):
    """
    Derives the effective cosmic energy density from the canonical partition function 
    of the MI-Laplacian spectrum, removing the ad-hoc spectral gap mapping.
    
    Z = Sum( e^(-beta * lambda_n) )
    E = -d(ln Z)/d(beta) = Sum( lambda_n * e^(-beta * lambda_n) ) / Z
    """
    # Exclude the zero-frequency translation ground state (lambda_0 = 0)
    active_modes = eigenvals[1:]
    
    # Compute partition weights
    exp_weights = np.exp(-beta * active_modes)
    Z = np.sum(exp_weights)
    
    # Internal thermal fluctuation energy density
    rho_eff = np.sum(active_modes * exp_weights) / Z
    return rho_eff

def simulate_saturation_sweep(L=8):
    N = L * L
    D = get_torus_distances(L)
    mask = D > 0
    
    # Test values approaching absolute saturation (alpha -> 0)
    alpha_sweep = [1.0, 0.5, 0.2, 0.05, 0.0]
    a_scale = 1.05  # Infinitesimal spatial volume deformation
    
    print(f"--- Sweeping Alpha down to Saturation Limit (Lattice N={N}) ---")
    print(f"--- Natively Derived via Canonical Partition Function ---")
    print(f"{'Alpha':<10}{'w_eff':<12}{'Thermodynamic Energy Ceiling':<30}{'Weyl Dimension'}")
    print("-" * 75)
    
    for alpha in alpha_sweep:
        # 1. Base Substrate Construction
        I_base = np.zeros_like(D)
        I_base[mask] = 1.0 / (D[mask] ** alpha)
        L_base = np.diag(np.sum(I_base, axis=1)) - I_base
        vals_base = np.sort(la.eigvalsh(L_base))
        
        # 2. Deformed Scale Construction
        I_scaled = np.zeros_like(D)
        I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha)
        L_scaled = np.diag(np.sum(I_scaled, axis=1)) - I_scaled
        vals_scaled = np.sort(la.eigvalsh(L_scaled))
        
        # 3. Statistical Mechanics Energy Extraction
        rho_base = compute_thermodynamic_energy(vals_base, beta=0.1)
        rho_scaled = compute_thermodynamic_energy(vals_scaled, beta=0.1)
        
        # 4. Parameter Calculations from First-Principles Energy Scaling
        log_rho_ratio = np.log(rho_scaled / rho_base)
        log_a_ratio = np.log(a_scale)
        scaling_exponent = log_rho_ratio / log_a_ratio
        w_eff = (-scaling_exponent / 3.0) - 1.0
        
        # 5. Dimension Slope Calculation
        max_mode = int(N * 0.25)
        modes = np.arange(1, max_mode)
        slope, _ = np.polyfit(np.log(modes), np.log(vals_base[modes]), 1)
        d_eff = 2.0 / slope if slope > 1e-10 else float('inf')
        
        print(f"{alpha:<10.3f}{w_eff:<12.4f}{rho_base:<30.4f}{f'{d_eff:.2f}D' if d_eff != float('inf') else 'InfiniteD'}")

if __name__ == "__main__":
    simulate_saturation_sweep(L=8)
