import torch
from torch.utils.data import Dataset
import numpy as np


class SpinChainDataset(Dataset):
    """
    Dataset representing a 1D Heisenberg Spin Chain.

    Physics:
    - Lattice of N spins (spin-1/2).
    - Hamiltonian H = -J * sum(S_i * S_{i+1}).
    - State is represented by the expectation values <Sz_i> or the full wavefunction amplitudes (simplified here to local magnetization vectors for efficiency).

    For a "Small" dataset efficient for DL training, we simulate:
    - Context: Random initial spin configuration (up/down with some domain walls).
    - Evolution: Time evolution under the Hamiltonian (approximated by nearest neighbor interaction updates).
    """

    def __init__(self, size: int = 10000, num_spins: int = 16, J: float = 1.0):
        """
        Args:
            size: Number of samples.
            num_spins: Number of sites in the chain (N).
            J: Coupling constant (Interaction strength).
        """
        self.size = size
        self.num_spins = num_spins
        self.J = J

        # Precompute interaction mask or sparse Hamiltonian logic if needed.
        # For <Sz> evolution, we need full quantum simulation or a simplified classical vector model.
        # "Classical Spin Chain" (O(3) vector model) is a good proxy for training complexity without 2^N state space.
        # Let's use Classical Heisenberg Model for O(N) generation speed vs O(2^N).
        # State: [N, 3] vectors (Sx, Sy, Sz).

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 1. Initialize random spin state (Classical vectors on sphere)
        # Shape: [N, 3] -> Flattened to [3N] or kept as [N, 3]
        # For compatibility with our model (expecting 1D vectors for now, or we reshape),
        # let's flatten to [N*3] input features.

        # Random angles theta, phi
        theta = torch.rand(self.num_spins) * np.pi
        phi = torch.rand(self.num_spins) * 2 * np.pi

        sx = torch.sin(theta) * torch.cos(phi)
        sy = torch.sin(theta) * torch.sin(phi)
        sz = torch.cos(theta)

        # State: [3, N]
        state = torch.stack([sx, sy, sz], dim=0)  # [3, N]

        # 2. Random Time Step
        dt = torch.rand(1) * 0.5  # Small steps to keep integration stable

        # 3. Integrate EOM (Equations of Motion)
        # dS_i/dt = S_i x (S_{i-1} + S_{i+1}) (Precession around local field)
        # We do a simple RK4 or Euler step.

        # Periodic Boundary Conditions
        # Neighbors sum:
        # Left neighbor: roll +1
        # Right neighbor: roll -1

        def compute_derivative(s):
            # s: [3, N]
            s_left = torch.roll(s, shifts=1, dims=1)
            s_right = torch.roll(s, shifts=-1, dims=1)
            local_field = self.J * (s_left + s_right)

            # Cross product S x B
            # Manual cross product for [3, N]
            # cx = sy*bz - sz*by
            # cy = sz*bx - sx*bz
            # cz = sx*by - sy*bx

            ds = torch.cross(s, local_field, dim=0)
            return ds

        # RK4 Integration
        k1 = compute_derivative(state)
        k2 = compute_derivative(state + 0.5 * dt * k1)
        k3 = compute_derivative(state + 0.5 * dt * k2)
        k4 = compute_derivative(state + dt * k3)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Renormalize to ensure unit length (physics constraint)
        next_state = next_state / (next_state.norm(dim=0, keepdim=True) + 1e-6)

        # Noise
        noise = torch.randn_like(next_state) * 0.05
        target_state = next_state + noise

        return {
            "x_context": state.flatten(),  # [3*N]
            "x_target": target_state.flatten(),  # [3*N]
            "dt": dt,  # [1]
        }
