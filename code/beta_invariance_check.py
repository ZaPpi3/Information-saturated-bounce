"""Beta-invariance check for the equation-of-state identity (Table I,
"Quantifying the finite-beta approximation").

w_eff = alpha/3 - 1 (Eq. 3) is an exact analytic identity only in the
beta->0+ limit, where the partition weights exp(-beta*lambda_n) degenerate
to uniform weights. At the finite beta=0.1 actually used throughout this
paper's figures, the weights pick up their own a-dependence through
lambda_n(a), so the identity is an approximation rather than exact away
from that limit. This script quantifies that approximation directly at a
reference point (alpha=1, exact w_eff=-2/3) across five orders of
magnitude in beta, on the same L=8 torus substrate (N=64) used elsewhere
in this paper.
"""
import numpy as np
import scipy.linalg as la


def get_torus_distances(L):
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    pos = np.vstack([x.ravel(), y.ravel()]).T
    N = L * L
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = min(abs(pos[i, 0] - pos[j, 0]), L - abs(pos[i, 0] - pos[j, 0]))
                dy = min(abs(pos[i, 1] - pos[j, 1]), L - abs(pos[i, 1] - pos[j, 1]))
                D[i, j] = np.sqrt(dx**2 + dy**2)
    return D


def rho_eff(eigenvals, beta):
    active = eigenvals[1:]
    if beta == 0:
        return active.mean()
    w = np.exp(-beta * (active - active.min()))  # shift for numerical stability, cancels in ratio
    return np.sum(active * w) / np.sum(w)


def w_eff_measured(D, mask, alpha, beta, a_scale=1.05):
    I_base = np.zeros_like(D)
    I_base[mask] = 1.0 / (D[mask] ** alpha) if alpha > 0 else 1.0
    vals_base = np.sort(la.eigvalsh(np.diag(np.sum(I_base, axis=1)) - I_base))
    rho_base = rho_eff(vals_base, beta)

    I_scaled = np.zeros_like(D)
    I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha) if alpha > 0 else 1.0
    vals_scaled = np.sort(la.eigvalsh(np.diag(np.sum(I_scaled, axis=1)) - I_scaled))
    rho_scaled = rho_eff(vals_scaled, beta)

    scaling_exponent = np.log(rho_scaled / rho_base) / np.log(a_scale)
    return -scaling_exponent / 3.0 - 1.0, rho_base


def main(L=8):
    D = get_torus_distances(L)
    mask = D > 0
    N = L * L
    betas = [0.001, 0.01, 0.1, 1.0, 10.0]
    alpha_ref = 1.0
    exact_w = alpha_ref / 3.0 - 1.0  # -0.6667

    print(f"Substrate: L={L} torus, N={N}. Reference alpha={alpha_ref}, exact w_eff={exact_w:.4f}")
    print(f"{'beta':<10}{'rho_eff(alpha=0)':<20}{'w_eff(alpha=1)':<18}{'|deviation| (rel.)'}")
    print("-" * 68)
    for beta in betas:
        w_a1, _ = w_eff_measured(D, mask, alpha_ref, beta)
        _, rho_a0 = w_eff_measured(D, mask, 1e-9, beta)  # alpha->0 (avoid exact 0 div issues)
        rel_dev = abs(w_a1 - exact_w) / abs(exact_w)
        print(f"{beta:<10.3f}{rho_a0:<20.4f}{w_a1:<18.4f}{rel_dev:.4%}")


if __name__ == "__main__":
    main()
