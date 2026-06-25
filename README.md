# Information-Saturated Bounce: Numerical Simulation Engine

This repository hosts the full numerical simulation ecosystem for an information-theoretic cosmology. The engine tracks the spectral properties of Mutual Information (MI) Laplacians on relational quantum substrates, demonstrating a universal de Sitter attractor, a hard ultraviolet energy ceiling, and topology-independent Renormalization Group (RG) flows.

## 📁 Repository Structure

```text
├── code/
│   ├── mi_laplacian_deflation.py        # Maps continuous scale-dependent dimension flows
│   ├── entanglement_cosmology_solvers.py # Computes finite-size scaling and strict AIC/BIC matrices
│   ├── saturation_bounce_plots.py       # Simulates equation of state transitions and energy limits
│   ├── universality_fss_engine.py       # Validates structural invariance across diverse graph families
│   ├── substrate_robustness_test.py     # Measures tolerance under localized edge-destruction defects
│   ├── rg_flow_trajectory_engine.py     # Computes the 2D cosmological vector field phase portrait
│   ├── rg_manifold_3d_engine.py         # Visualizes the unified 3D multivariable RG manifold
│   └── rg_stability_analysis.py         # Smoothes trajectories and extracts universal critical exponents
├── figures/                             # High-fidelity publication-grade PNG visualizations
├── main.tex                             # Production LaTeX manuscript source
├── main.pdf                             # Compiled preprint manuscript
└── LICENSE                              # MIT Open-Source License
```

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

### 1. Metric Analysis & Information Criterion Validation
To run the primary finite-size lattice scaling loops and output the strict information-theoretic selection metrics (AIC/BIC scores):
```bash
python code/entanglement_cosmology_solvers.py
```

### 2. Multi-Topology Universality Diagnostics
To test the framework's complete independence from background coordinates by running sweeps across Erdős–Rényi, Watts–Strogatz, and Barabási–Albert graph families:
```bash
python code/universality_fss_engine.py
```

### 3. Extracting the Renormalization Group Exponents
To execute the centered finite-difference stencils, smooth out discrete grid artifacts via Savitzky-Golay filters, and print out the universal linearized critical exponents ($A$ and $B$):
```bash
python code/rg_stability_analysis.py
```

## 🧠 Core Physics Framework

This numerical architecture simulates the structural dynamics of a cosmic crunch transitioning into an inflationary bounce. By driving the network substrate down to the absolute information-saturation limit ($\alpha \rightarrow 0^+$), the code records three universal properties of the quantum substrate:

1. **The Repulsive Switch ($w_{\text{eff}} \rightarrow -1.0$)**: Scale deformations fail to dilute matrix weights when distance dependencies collapse. The effective energy density freezes ($\rho_{\text{eff}} \propto a^0$), activating an instantaneous cosmological constant attractor state.
2. **The Ultraviolet Energy Ceiling ($\lambda_1 = N$)**: Rather than diverging to infinity ($\rho \rightarrow \infty$), the fundamental spectral gap locks onto a rigid algebraic maximum bounded by the total system degrees of freedom. This acts as an inherent geometric shock absorber.
3. **Universal Attractor Funneling**: The 3D phase space maps out a highly focused trajectory where widely disparate initial network microstates lose all historical memory, collapsing onto a single low-dimensional manifold tracking toward a fixed fractional dimension of $d_s \approx 5.25$.

## 📜 Licensing

This software engine is licensed under the open-source **MIT License**. Feel free to use, modify, and distribute the pipelines with appropriate attribution.
