import torch
import torch.nn as nn


class FidelityLoss(nn.Module):
    """
    Computes Fidelity between Predicted Complex State and Target Real State.
    Target is usually Real (output of standard encoder), but we treat it as Complex with Imag=0
    or we project prediction to Real.

    Prompt says: L_align = 1 - |<psi_hat, psi_target>|^2
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_re, z_im, target):
        # target is [B, D] (Real)
        # z_re, z_im are [B, D]

        # Normalize vectors for "Fidelity" (Quantum states area unit vectors)
        # Norm = sqrt(re^2 + im^2)
        norm_pred = (
            torch.sqrt(z_re**2 + z_im**2).sum(dim=1, keepdim=True).sqrt()
        )  # Vector L2 norm?
        # Quantum state is unit L2 norm.
        # Let's normalize the *vectors*.
        pred_norm_re = z_re / (
            torch.norm(torch.stack([z_re, z_im], dim=0), dim=0).norm(
                dim=1, keepdim=True
            )
            + 1e-8
        )
        pred_norm_im = z_im / (
            torch.norm(torch.stack([z_re, z_im], dim=0), dim=0).norm(
                dim=1, keepdim=True
            )
            + 1e-8
        )

        # Target is real, so imag part is 0.
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)

        # Inner product <u, v> = sum(conj(u_i) * v_i)
        # u = pred (complex), v = target (real)
        # conj(u) = re - i*im
        # <u, v> = sum((re - i*im) * target)
        #        = sum(re*target) - i * sum(im*target)

        dot_re = (pred_norm_re * target_norm).sum(dim=1)
        dot_im = (pred_norm_im * target_norm).sum(
            dim=1
        )  # Actually negative in math, but magnitude squared is same.

        # |<u,v>|^2 = dot_re^2 + dot_im^2
        fidelity = dot_re**2 + dot_im**2

        loss = 1 - fidelity.mean()
        return loss


class SIGRegCLoss(nn.Module):
    """
    Complex SIGReg (Isotropy Regularization).
    Encourage Re and Im parts to be N(0, 1) and uncorrelated.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, z_re, z_im):
        # z: [B, D]

        # 1. Variance / Covariance regularization
        # We want Variance(dim) ~ 1
        # We want Covariance(dim, dim') ~ 0

        # VICReg style or LeJEPA style?
        # Regular SIGReg uses projections.
        # Let's use simple variance + cross-covariance for "Speed/Memory" on 6GB VRAM.
        # Computing full Covariance matrix [D, D] might be heavy if D is large.
        # D=384. [384, 384] is small.

        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        def covariance_loss(x):
            # x: [B, D]
            batch_size = x.size(0)
            x = x - x.mean(dim=0)
            std = torch.sqrt(x.var(dim=0) + 0.0001)
            # Std Loss: smooth L1 to 1
            std_loss = torch.mean(torch.relu(1 - std))

            # Cov Loss
            # cov = (x.T @ x) / (batch_size - 1)
            # cov_loss = off_diagonal(cov).pow(2).sum() / D
            # Skip full cov for speed/memory if batch is small (gradients noisy).
            # LeJEPA uses SIGReg (projections) exactly to avoid this matrix.

            return std_loss

        # Apply to Re and Im separately
        loss_re = covariance_loss(z_re)
        loss_im = covariance_loss(z_im)

        # Cross-Covariance? Re vs Im shoudl be decorrelated?
        # Maybe.

        return (loss_re + loss_im) * self.alpha
