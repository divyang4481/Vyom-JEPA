### 2. ARCHITECTURE.md

_The technical "Deep Dive" into the math and system design._

```markdown
# System Architecture: Vyom-JEPA

## 1. Mathematical Foundation

Vyom-JEPA operates in a Latent-Euclidean space $\mathbb{R}^d$.

### The JEPA Components

1. **Context Encoder ($E_\theta$):** A Transformer-based map $x \rightarrow z$.
2. **Predictor ($P_\psi$):** A narrow latent-bridge that predicts $z_{target}$ from $z_{context}$ given a positional/time-step prompt.
3. **Target Encoder ($E_\phi$):** An Exponential Moving Average (EMA) version of $E_\theta$.

### Loss Function: SIGReg

To ensure the embeddings are spread uniformly across the "Vyom" (latent space), we implement **Sketched Isotropic Gaussian Regularization**:
$$L = \text{MSE}(\hat{z}_y, z_y) + \alpha \cdot L_{var} + \beta \cdot L_{cov}$$

- **Variance ($L_{var}$):** Forces each latent dimension to have a standard deviation near 1.
- **Covariance ($L_{cov}$):** Penalizes off-diagonal elements in the covariance matrix to ensure dimensions are uncorrelated.

## 2. Quantum Integration

- **Input Space:** Fock states represented as occupancy tensors $|n_1, n_2, \dots, n_k\rangle$.
- **Evolution:** The Predictor module simulates the action of a Hamiltonian $\hat{H}$ in the latent space.

## 3. Hardware Optimization

- **Precision:** Mixed-precision (FP16) for 6GB VRAM compatibility.
- **Bottleneck:** Use of a "Lean" Latent dimension ($d=128$ or $d=256$) to minimize memory overhead.
```
