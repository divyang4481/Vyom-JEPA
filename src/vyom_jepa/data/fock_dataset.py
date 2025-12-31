import torch
from torch.utils.data import Dataset
import numpy as np


class FockDynamicsDataset(Dataset):
    """
    Dataset generating Fock state dynamics.

    Data:
    - Input: Occupancy vectors |n1, n2, ..., nk> (approximated as float vectors for this continuous-variable task).
    - Dynamics: Unitary-inspired transformation + Noise.
    """

    def __init__(self, size: int = 10000, num_modes: int = 16, max_occupation: int = 5):
        """
        Args:
            size: Number of samples to generate per epoch (or fixed pool size).
            num_modes: Dimension of the Fock state vector (k).
            max_occupation: Max value for initialization (n).
        """
        self.size = size
        self.num_modes = num_modes
        self.max_occupation = max_occupation

        # Pre-generate some random unitary-like matrices (rotations) to simulate physics
        # We can construct a random skew-symmetric matrix H and use exp(H * dt) to get orthogonal evolution
        # For simplicity in 'getitem', we might generate dynamics on the fly or use a fixed Hamiltonian.
        # Let's use a fixed random Hamiltonian for consistent physics.
        self.hamiltonian = torch.randn(num_modes, num_modes)
        self.hamiltonian = (
            self.hamiltonian - self.hamiltonian.T
        )  # Skew-symmetric -> Orthogonal evolution

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 1. Generate Context State (Initial Fock State)
        # Random integers, then normalized or noise added.
        # The prompt implies continuous "Occupancy vectors", so we'll start with integers and convert to float.
        # Shape: [num_modes]
        x_context_int = torch.randint(
            0, self.max_occupation + 1, (self.num_modes,)
        ).float()

        # Normalize to keep values reasonable for NN
        # In Fock physics, norm is particle number, but for NN inputs we usually want std roughly 1.
        # Let's just pass raw values or slightly scaled.
        x_context = x_context_int

        # 2. Pick a random time step 'dt' (action)
        dt = torch.rand(1) * 2.0  # Time between 0 and 2.0

        # 3. Apply Evolution: x_target = exp(H * dt) @ x_context
        # Approximation: Matrix exponential.
        # U = torch.matrix_exp(self.hamiltonian * dt) is expensive to do per item if K is large.
        # For small K (16), it's fast.

        # Note: We need to do this computation.
        # Since 'dt' varies per sample, we can't precompute U.
        # Optimization: matrix_exp on CPU/GPU.

        U = torch.matrix_exp(self.hamiltonian * dt.item())
        x_pure_target = U @ x_context

        # 4. Add small Gaussian noise
        noise = torch.randn_like(x_pure_target) * 0.1
        x_target = x_pure_target + noise

        # Return dictionary
        return {
            "x_context": x_context,  # [K]
            "x_target": x_target,  # [K]
            "dt": dt,  # [1]
        }
