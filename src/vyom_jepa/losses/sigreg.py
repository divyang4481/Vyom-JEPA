import torch
import torch.nn as nn
import numpy as np

class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg) Loss.
    
    Implements the "Simplified Characteristic Function loss" to force 
    random projections of the latent space to match a Standard Normal distribution N(0, 1).
    
    Reference: "Provable SSL Without Heuristics" (2025) - LeJEPA
    """
    def __init__(self, feature_dim: int, num_projections: int = 16, t_points: int = 10, sigma: float = 1.0):
        """
        Args:
            feature_dim: Dimension of the latent space z.
            num_projections: Number of random 1D directions (M) to project onto.
            t_points: Number of points to evaluate the Characteristic Function difference.
            sigma: Scale for the range of t to evaluate (t ~ U[-3sigma, 3sigma]).
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_projections = num_projections
        self.t_points = t_points
        self.sigma = sigma
        
        # Buffer for random projection matrix W. 
        # In strictly "random" projections, this might be fixed. 
        # For "Sketched", we often resample or use a fixed set.
        # Fixed random matrix is more stable for optimization.
        self.register_buffer('projections', torch.randn(feature_dim, num_projections))
        
        # Points at which to evaluate the characteristic function (CF).
        # We sample t around 0 where the CF is most informative.
        self.register_buffer('t_eval', torch.linspace(-3*sigma, 3*sigma, t_points))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent representations (Batch, Dim) or (Batch, Seq, Dim).
               If sequence, we flatten or average. Use flatten for strict distribution control.
        
        Returns:
            Scalar loss value.
        """
        # Ensure z is flat: (Batch * Seq, Dim) or (Batch, Dim)
        if z.dim() > 2:
            z = z.flatten(0, 1)
            
        # 1. Project latent z onto M random directions
        # z: [B, D], W: [D, M] -> projected: [B, M]
        # Normalize projections vectors to unit length to ensure "Standard" Normal scaling
        w = self.projections / (self.projections.norm(dim=0, keepdim=True) + 1e-6)
        projections = z @ w # [B, M]
        
        # 2. Compute Empirical CF for each projection direction m
        # phi_hat(t) = (1/N) * sum_{j=1}^N exp(i * t * p_{j,m})
        # We evaluate this at several t points.
        
        # t_eval: [T]
        # projections: [B, M]
        # We want result of shape [M, T] (CF for each projection at each t)
        
        # Expand for broadcasting:
        # t: [1, 1, T]
        # p: [B, M, 1]
        t = self.t_eval.view(1, 1, -1)
        p = projections.unsqueeze(-1)
        
        # argument: p * t -> [B, M, T]
        args = p * t
        
        # cos and sin components
        cos_sum = torch.cos(args).mean(dim=0) # [M, T]
        sin_sum = torch.sin(args).mean(dim=0) # [M, T]
        
        # 3. Analytic CF for N(0, 1) is exp(-0.5 * t^2)
        # It's real-valued (symmetric distribution).
        t_sq = self.t_eval ** 2
        
        # Target real part: exp(-0.5 * t^2)
        # Target imag part: 0
        target_re = torch.exp(-0.5 * t_sq).unsqueeze(0).expand(self.num_projections, -1) # [M, T]
        target_im = torch.zeros_like(target_re)
        
        # 4. Squared L2 distance between empirical and target CF
        loss_re = (cos_sum - target_re) ** 2
        loss_im = (sin_sum - target_im) ** 2
        
        loss = (loss_re + loss_im).mean()
        
        return loss
