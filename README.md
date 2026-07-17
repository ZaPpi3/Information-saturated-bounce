# Information-Saturated Bounce: Numerical Simulation Engine

This repository hosts the full numerical simulation ecosystem for an information-theoretic cosmology. The engine tracks the spectral properties of Mutual Information (MI) Laplacians on relational quantum substrates, demonstrating a universal de Sitter attractor, a hard ultraviolet energy ceiling, and topology-independent Renormalization Group (RG) flows.

## 📁 Repository Structure

```text
├── code/
│   ├── saturation_bounce_check.py       # Equation-of-state / energy-ceiling sweep (Figs. 1-2)
│   ├── beta_invariance_check.py         # Quantifies the finite-beta approximation (Table I)
│   ├── saturation_thermodynamic_scaling.py # Finite-size scaling of the alpha=0 ceiling
│   ├── universality_fss_engine.py       # Structural invariance across graph families
│   ├── substrate_robustness_test.py     # Tolerance under localized edge-destruction defects
│   ├── attractor_stability_stress_test.py # Attractor stability under perturbation
│   ├── rg_flow_trajectory_engine.py     # 2D cosmological vector field phase portrait
│   ├── rg_manifold_3d_engine.py         # Unified 3D multivariable RG manifold
│   ├── rg_beta_function_solver.py       # RG beta-function extraction
│   ├── rg_stability_analysis.py         # Smooths trajectories, extracts single-config exponents (superseded, see below)
│   ├── spectral_dimension_rg_flow.py    # Spectral dimension along the RG flow
│   └── critical_exponent_finite_size_scaling.py # Finite-size sweep: shows A/B aren't universal, replaces them with a stable pooled exponent p
├── figures/                             # PNG figures referenced by main.tex
├── main.tex                             # Manuscript source
├── main.pdf                             # Compiled manuscript
└── LICENSE                              # MIT License
```

Note: this list previously named files (`mi_laplacian_deflation.py`,
`entanglement_cosmology_solvers.py`, `saturation_bounce_plots.py`) that do
not exist anywhere in this repository. Corrected to match the actual
`code/` directory.

## ⚙️ Installation & System Requirements

The computational core requires Python 3.9+ along with standard open-source scientific computing and visualization libraries. 

Clone the repository and install the verified dependencies using your package manager:

```bash
git clone https://github.com
cd Information-saturated-bounce
pip install numpy scipy matplotlib
```

## 🚀 Execution Pipelines

Every simulator is self-contained and outputs high-fidelity visuals directly into the `figures/` directory.

### 1. Equation of state, energy ceiling, and the finite-$\beta$ check
```bash
python code/saturation_bounce_check.py       # Figs. 1-2
python code/beta_invariance_check.py         # Table I
python code/saturation_thermodynamic_scaling.py
```

### 2. Multi-Topology Universality Diagnostics
To test the framework's complete independence from background coordinates by running sweeps across Erdős–Rényi, Watts–Strogatz, and Barabási–Albert graph families:
```bash
python code/universality_fss_engine.py
```

### 3. Extracting the Renormalization Group Exponents
`rg_stability_analysis.py` runs the centered finite-difference + Savitzky-Golay procedure at a single configuration (N=100, Barabási-Albert), printing $A$ and $B$:
```bash
python code/rg_stability_analysis.py
```
These are **not universal** - see the next script. `critical_exponent_finite_size_scaling.py` repeats the same procedure across $N=50,100,200,400$ and all three topologies, shows $A$/$B$ vary by orders of magnitude (and $B$ changes sign) rather than converging, and fits the more robust pooled near-boundary power law $1+w_{\text{eff}} = C(1-\lambda_1/N)^p$ instead, which **is** stable ($p = 1.122 \pm 0.008$ across the same sweep):
```bash
python code/critical_exponent_finite_size_scaling.py
```

## 🧠 Core Physics Framework

This numerical architecture simulates the structural dynamics of a cosmic crunch transitioning into an inflationary bounce. By driving the network substrate down to the absolute information-saturation limit ($\alpha \rightarrow 0^+$), the code records three universal properties of the quantum substrate:

1. **The Repulsive Switch ($w_{\text{eff}} \rightarrow -1.0$)**: Scale deformations fail to dilute matrix weights when distance dependencies collapse. The effective energy density freezes ($\rho_{\text{eff}} \propto a^0$), activating an instantaneous cosmological constant attractor state.
2. **The Ultraviolet Energy Ceiling ($\lambda_1 = N$)**: Rather than diverging to infinity ($\rho \rightarrow \infty$), the fundamental spectral gap locks onto a rigid algebraic maximum bounded by the total system degrees of freedom. This acts as an inherent geometric shock absorber.
3. **Universal Attractor Funneling**: The 3D phase space maps out a highly focused trajectory where widely disparate initial network microstates lose all historical memory, collapsing onto a single low-dimensional manifold. The fractional dimension at this manifold's plateau ($d_s \approx 5.25$ at this paper's reference size $N=100$) is *not* itself universal - it grows with $N$ (see `critical_exponent_finite_size_scaling.py`); the reproducible size-stable quantity is instead the critical exponent $p = 1.122 \pm 0.008$ governing the approach to the fixed point.

## 📜 Licensing

This software engine is licensed under the open-source **MIT License**. Feel free to use, modify, and distribute the pipelines with appropriate attribution.
