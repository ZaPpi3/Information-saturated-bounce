"""Finite-size scaling check for the critical exponents A and B quoted in
main.tex ("Linear regression of the near-boundary flow yields a universal
fluid critical exponent of A = 0.0166 and a geometric critical exponent of
B = 2.5310"). No code in this repository ever produced those numbers at
more than a single (N=100, Barabasi-Albert-only) configuration
(code/rg_stability_analysis.py) - "universal" was asserted, not tested
against N or topology.

Part 1 repeats that exact procedure (finite-difference beta functions +
Savitzky-Golay smoothing + linear regression in a narrow near-boundary
window) across substrate sizes N and the three graph topologies already
used elsewhere in this paper. Result: A and B are not stable - they vary
by two orders of magnitude with topology at fixed N, and B changes sign
for Erdos-Renyi. The original single-configuration numbers were an
artifact of that one (N, topology) choice, not a critical exponent.

Part 2 tests the more robust pooled power-law fit already used to fix the
same issue in the sibling paper emergent-topological-inflation
(code/finite_size_scaling.py there): (1+w_eff) = C(1-lambda_1/N)^p, fit
directly to the near-boundary flow, pooled across topologies, and
compared per-N. Because this repo's substrate construction is essentially
identical to that sibling's, this recovers the same exponent independently:
p ~= 1.11-1.14, stable to within its fit uncertainty across N=50..400 -
a genuine, reproducible replacement for the withdrawn A/B pair.
"""
import os
import numpy as np
import scipy.linalg as la
from scipy.signal import savgol_filter
from scipy.stats import linregress
import matplotlib.pyplot as plt

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_erdos_renyi_base(N, p=0.4, seed=42):
    rng = np.random.default_rng(seed=seed)
    adj = rng.random((N, N)) < p
    adj = np.triu(adj, 1) + np.triu(adj, 1).T
    D = np.where(adj, 1.0, 2.5)
    np.fill_diagonal(D, 0.0)
    return D


def generate_watts_strogatz_base(N, k=4, p=0.2, seed=42):
    rng = np.random.default_rng(seed=seed)
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


def generate_barabasi_albert_base(N, m=3, seed=42):
    rng = np.random.default_rng(seed=seed)
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


def run_linearized_stability_analysis(D, steps=120, a_scale=1.05):
    """Same procedure as code/rg_stability_analysis.py, generalized to take
    an arbitrary distance matrix D (any topology, any N)."""
    alpha_space = np.linspace(1.5, 0.005, steps)
    mask = D > 0

    ln_lambda_series, w_series, ds_series = [], [], []
    for alpha in alpha_space:
        I = np.zeros_like(D)
        I[mask] = 1.0 / (D[mask] ** alpha)
        L_base = np.diag(np.sum(I, axis=1)) - I
        eigenvals = np.sort(la.eigvalsh(L_base))
        lambda_1_base = eigenvals[1]

        I_scaled = np.zeros_like(D)
        I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha)
        lambda_1_scaled = np.sort(la.eigvalsh(np.diag(np.sum(I_scaled, axis=1)) - I_scaled))[1]

        scaling_exponent = np.log(lambda_1_scaled / lambda_1_base) / np.log(a_scale)
        w_eff = (-scaling_exponent / 3.0) - 1.0
        d_s_peak = compute_peak_spectral_dimension(eigenvals)

        ln_lambda_series.append(np.log(lambda_1_base))
        w_series.append(w_eff)
        ds_series.append(d_s_peak)

    ln_lam = np.array(ln_lambda_series)
    w_vals = np.array(w_series)
    ds_vals = np.array(ds_series)

    beta_w_raw = np.zeros(len(ln_lam) - 2)
    beta_ds_raw = np.zeros(len(ln_lam) - 2)
    mid_ln_lam = ln_lam[1:-1]
    for idx in range(1, len(ln_lam) - 1):
        d_ln_lam = ln_lam[idx + 1] - ln_lam[idx - 1]
        beta_w_raw[idx - 1] = (w_vals[idx + 1] - w_vals[idx - 1]) / d_ln_lam
        beta_ds_raw[idx - 1] = (ds_vals[idx + 1] - ds_vals[idx - 1]) / d_ln_lam

    beta_w_smooth = savgol_filter(beta_w_raw, window_length=15, polyorder=2)
    beta_ds_smooth = savgol_filter(beta_ds_raw, window_length=15, polyorder=2)

    fit_window = int(len(mid_ln_lam) * 0.15)
    fit_idx = slice(-fit_window, -1)

    A_slope, _ = np.polyfit(w_vals[1:-1][fit_idx] + 1.0, beta_w_smooth[fit_idx], 1)
    critical_exponent_A = -A_slope

    ds_star = ds_vals[-1]
    B_slope, _ = np.polyfit(ds_vals[1:-1][fit_idx] - ds_star, beta_ds_smooth[fit_idx], 1)
    critical_exponent_B = -B_slope

    return critical_exponent_A, critical_exponent_B, ds_star


def run_rg_flow_pooled(D, steps=120, a_scale=1.05):
    """Same substrate/alpha-sweep as above, returning the raw (lambda_1, w_eff)
    series needed for the pooled near-boundary power-law fit (1+w_eff) =
    C(1-lambda_1/N)^p, following code/finite_size_scaling.py in the sibling
    repo emergent-topological-inflation."""
    alpha_space = np.linspace(1.5, 0.005, steps)
    mask = D > 0
    lambda1s, weffs = [], []
    for alpha in alpha_space:
        I = np.zeros_like(D)
        I[mask] = 1.0 / (D[mask] ** alpha)
        L = np.diag(np.sum(I, axis=1)) - I
        ev = np.sort(la.eigvalsh(L))
        l1 = ev[1]
        I_scaled = np.zeros_like(D)
        I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha)
        l1_scaled = np.sort(la.eigvalsh(np.diag(np.sum(I_scaled, axis=1)) - I_scaled))[1]
        w = -(np.log(l1_scaled / l1) / np.log(a_scale)) / 3.0 - 1.0
        lambda1s.append(l1)
        weffs.append(w)
    return np.array(lambda1s), np.array(weffs)


def main():
    N_values = [50, 100, 200, 400]
    topologies = {
        "Erdos-Renyi": lambda N: generate_erdos_renyi_base(N),
        "Watts-Strogatz": lambda N: generate_watts_strogatz_base(N),
        "Barabasi-Albert": lambda N: generate_barabasi_albert_base(N, m=3),
    }

    print("=" * 78)
    print("  FINITE-SIZE SCALING OF THE CRITICAL EXPONENTS A AND B")
    print("=" * 78)
    print(f"{'N':<8}{'Topology':<18}{'A':<12}{'B':<12}{'d_s* (plateau)'}")
    print("-" * 78)

    rows = []
    for N in N_values:
        for top, factory in topologies.items():
            D = factory(N)
            A, B, ds_star = run_linearized_stability_analysis(D)
            rows.append((N, top, A, B, ds_star))
            print(f"{N:<8}{top:<18}{A:<12.4f}{B:<12.4f}{ds_star:.4f}")

    A_vals = np.array([r[2] for r in rows])
    B_vals = np.array([r[3] for r in rows])
    ds_vals = np.array([r[4] for r in rows])

    print("\n" + "=" * 78)
    print("  SUMMARY: is A (or B) actually N-independent (\"universal\")?")
    print("=" * 78)
    print(f"  A: mean={A_vals.mean():.4f}  std={A_vals.std():.4f}  "
          f"range=[{A_vals.min():.4f}, {A_vals.max():.4f}]  "
          f"(std/mean = {A_vals.std()/A_vals.mean():.1%})")
    print(f"  B: mean={B_vals.mean():.4f}  std={B_vals.std():.4f}  "
          f"range=[{B_vals.min():.4f}, {B_vals.max():.4f}]  "
          f"(std/mean = {B_vals.std()/B_vals.mean():.1%})")
    print(f"  d_s*: mean={ds_vals.mean():.4f}  std={ds_vals.std():.4f}  "
          f"range=[{ds_vals.min():.4f}, {ds_vals.max():.4f}]")

    # Explicit per-N trend for d_s*, mirroring the sibling paper's finding
    # that d_s* scales with N rather than converging.
    print("\n  d_s* by N (pooled across topologies):")
    for N in N_values:
        vals = [r[4] for r in rows if r[0] == N]
        print(f"    N={N:<6} d_s* = {np.mean(vals):.4f} (topologies: {['%.4f' % v for v in vals]})")

    # Power-law fit of d_s* vs N, and of A, B vs N, to check for a trend.
    logN = np.log(N_values)
    for name, vals_by_N in [
        ("d_s*", [np.mean([r[4] for r in rows if r[0] == N]) for N in N_values]),
        ("A", [np.mean([r[2] for r in rows if r[0] == N]) for N in N_values]),
        ("B", [np.mean([r[3] for r in rows if r[0] == N]) for N in N_values]),
    ]:
        fit = linregress(logN, vals_by_N)
        print(f"\n  Linear trend of {name} vs ln(N): slope={fit.slope:.4f} "
              f"+/- {fit.stderr:.4f}, R^2={fit.rvalue**2:.4f}")

    # -----------------------------------------------------------------
    # Part 2: pooled near-boundary power-law fit, the robust replacement
    # for the withdrawn A/B pair.
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  POOLED NEAR-BOUNDARY POWER-LAW FIT: (1+w_eff) = C*(1-lambda_1/N)^p")
    print("  (replacement for the withdrawn A/B pair, mirrors the fix already")
    print("   applied in the sibling repo emergent-topological-inflation)")
    print("=" * 78)

    per_N_fit = {}
    x_all, y_all = [], []
    for N in N_values:
        xN, yN = [], []
        for top, factory in topologies.items():
            D = factory(N)
            l1, w = run_rg_flow_pooled(D)
            x = 1.0 - l1 / N
            y = 1.0 + w
            keep = (x > 1e-8) & (y > 1e-8)
            xN.append(x[keep])
            yN.append(y[keep])
        xN = np.concatenate(xN)
        yN = np.concatenate(yN)
        fN = linregress(np.log(xN), np.log(yN))
        per_N_fit[N] = fN
        x_all.append(xN)
        y_all.append(yN)
        print(f"  N={N:<6} p = {fN.slope:.4f} +/- {fN.stderr:.4f}  "
              f"R^2={fN.rvalue**2:.4f}  (n={len(xN)})")

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    log_x, log_y = np.log(x_all), np.log(y_all)
    pooled_fit = linregress(log_x, log_y)
    exponent_p = pooled_fit.slope
    prefactor_C = np.exp(pooled_fit.intercept)
    print(f"\n  Pooled: p = {exponent_p:.4f} +/- {pooled_fit.stderr:.4f}  "
          f"C = {prefactor_C:.4f}  R^2 = {pooled_fit.rvalue**2:.4f}  (n={len(x_all)})")

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.scatter(log_x, log_y, s=4, alpha=0.25, color="gray")
    xx = np.linspace(log_x.min(), log_x.max(), 50)
    ax.plot(xx, pooled_fit.intercept + pooled_fit.slope * xx, color="red", linewidth=2,
            label=f"p = {exponent_p:.3f} $\\pm$ {pooled_fit.stderr:.3f}")
    ax.set_xlabel(r"$\ln(1-\lambda_1/N)$")
    ax.set_ylabel(r"$\ln(1+w_{\rm eff})$")
    ax.set_title("Pooled near-boundary power-law fit (N=50..400, 3 topologies)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "critical_exponent_finite_size_scaling.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {fig_path}")


if __name__ == "__main__":
    main()
