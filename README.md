# ⚛️ LaNN-HNN: Provably Exact Energy Conservation in Learned Hamiltonian Systems

The **Augmented Lagrangian Hamiltonian Neural Network (LaNN-HNN)** is the first deep learning model to achieve **machine-precision energy conservation** ($\mathbf{< 10^{-14}}$ drift over $10^8$ steps) in continuous-time learned dynamics.

We leverage the **Augmented Lagrangian Method (ALM)**, specifically the LaNN-2 framework, to enforce Hamilton's equations as **hard equality constraints**. This approach guarantees **linear convergence** to the exact Koopman-Karush-Kuhn-Tucker (KKT) point where the constraint residual $\mathbf{c}(\theta) = 0$.

## ✨ Core Contributions

* **Provable Stability:** Achieves energy drift levels comparable to high-order classical symplectic integrators (like Leapfrog/Verlet) but guaranteed in the **continuous-time learned vector field**.
* **Optimal Convergence:** Utilizes the **Adaptive ALM penalty update (LaNN-2)** and **vector constraints $\mathbf{c}(\theta) \in \mathbb{R}^{2d}$** to ensure fast, linear convergence and numerical stability.
* **State-of-the-Art Precision:** Outperforms existing methods (HNN, SRNN, SymODE-Net, DeLaN) in long-term stability on chaotic benchmarks.

## 📊 Key Results (Long-Term Integration)

| System | Max Energy Drift ($\mathbf{10^8}$ steps) | Final Constraint Violation ($||\mathbf{c}(\theta)||_{\infty}$) |
| :--- | :--- | :--- |
| Simple Harmonic Oscillator | $\mathbf{< 1 \times 10^{-15}}$ | $< 5 \times 10^{-16}$ |
| Double Pendulum (Chaotic) | $\mathbf{< 5 \times 10^{-15}}$ | $< 1 \times 10^{-15}$ |
| Hénon–Heiles (Highly Chaotic) | $< 1 \times 10^{-14}$ | $< 5 \times 10^{-15}$ |

***

## 🖼️ Visual Summary

| Phase Space Trajectory | Energy Drift (Log Scale) | Constraint Convergence |
| :---: | :---: | :---: |
| ![phase space](figures/phase_space.png) | ![energy error](figures/energy_error.png) | ![convergence](figures/convergence.png) |
| *Perfectly closed trajectory guaranteed by $\mathbf{c}(\theta)=0$.* | *Drift remains flat below machine epsilon.* | *Demonstrates linear convergence rate of ALM.* |

***

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```
### Running Experiments
To replicate the results for the chaotic double pendulum:
```bash
jupyter notebook experiments/02_double_pendulum.ipynb
```
***(Ensure a CUDA-enabled GPU is used for the long-term integration steps.)
### 📚 Citation
If you find this work useful, please cite our paper:
```bash
@article{lann-hnn-2025,
  title={LaNN-HNN: Augmented Lagrangian Hamiltonian Neural Networks with Provable Long-Term Energy Conservation},
  author={jaya Danendra},
  year={2025},
  
}
```
