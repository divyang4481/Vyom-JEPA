import torch
import torch.nn as nn
import copy


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Simple wrapper for Transformer Encoder"""

    def __init__(self, d_model, n_head, num_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class VyomJEPA(nn.Module):
    def __init__(
        self,
        input_dim=16,
        d_model=256,
        n_heads=8,
        n_layers=4,
        predictor_layers=2,
        use_ema=True,
        ema_decay=0.996,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 1. Encoders
        # Input projection: Map input dimension (16) to d_model (256)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Context Encoder
        self.context_encoder = TransformerBlock(d_model, n_heads, n_layers)

        # Target Encoder
        # If shared weights, we just point to context_encoder (conceptually).
        # But usually we want a separate object to handle EMA easily.
        self.target_encoder = copy.deepcopy(self.context_encoder)

        # If not using EMA (LeJEPA 'Proven' style often implies shared or specific constraints),
        # but prompt says "Support both... via config".
        # We start target_encoder with same weights.

        if self.use_ema:
            # Stop gradient for target encoder
            for p in self.target_encoder.parameters():
                p.requires_grad = False

        # 2. Predictor
        # Takes z_context + action.
        # Action is 'dt' (scalar). We embed it.
        self.action_embedding = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        self.predictor = TransformerBlock(d_model, n_heads, predictor_layers)

        # Output projection (optional, if we want to map back or just keep in latent z)
        # Latent prediction is usually done in the latent space itself.
        # But if we compare z_pred and z_target, they match dimension.

    def forward_context(self, x):
        # x: [B, K]
        # Transformer expects [B, Seq, D].
        # We can treat the K dimensions as 1 token or K tokens?
        # "Context Encoder (E_theta): A ViT or Transformer block mapping Fock occupancy vectors to latent z."
        # If input is [B, 16], let's project to [B, 1, D] and treat as single token
        # OR project to [B, 16, D] (treating each mode as a token? No, input is vector).
        # We will treat the whole state as one token for simplicity, or project to [B, 1, D].

        x_emb = self.input_proj(x).unsqueeze(1)  # [B, 1, D]
        z = self.context_encoder(x_emb)  # [B, 1, D]
        return z.squeeze(1)  # [B, D]

    def forward_target(self, x):
        # If Shared Weights and NOT EMA, we use context_encoder
        if not self.use_ema:
            return self.forward_context(x)

        with torch.no_grad():
            x_emb = self.input_proj(x).unsqueeze(1)
            z = self.target_encoder(x_emb)
            return z.squeeze(1)

    def forward(self, x_context, x_target, dt):
        """
        Returns:
            z_pred: Predicted latent representation from context + action
            z_target: Actual latent representation of target
        """
        # 1. Compute Context Latent
        z_c = self.forward_context(x_context)  # [B, D]

        # 2. Compute Target Latent
        z_t = self.forward_target(x_target)  # [B, D]

        # 3. Predictor
        # Combine z_c and action
        action_emb = self.action_embedding(dt)  # [B, D]

        # We feed sequence [z_c, action] or sum them?
        # JEPA usually conditions predictor on context.
        # Simple interactions: Add them or concat.
        # Prompt: "taking z_context + ... embedding" -> suggests addition or concatenation.
        # Let's interact them via Cross Attention or just separate tokens.
        # Let's treat them as sequence: [z_c_token, action_token] -> Predictor -> [z_pred_token]
        # Or just sum if it's a "Hamiltonian Action" implying time evolution operator.
        # e^{-iHt} applied to z.
        # Let's use sequence: [z_c] + [action]

        z_c_seq = z_c.unsqueeze(1)  # [B, 1, D]
        act_seq = action_emb.unsqueeze(1)  # [B, 1, D]

        pred_input = torch.cat([z_c_seq, act_seq], dim=1)  # [B, 2, D]
        z_pred_seq = self.predictor(pred_input)  # [B, 2, D]

        # We want to predict the state at the next step.
        # Which token corresponds to the state?
        # Usually the first one (corresponding to z_c evolved).
        z_pred = z_pred_seq[:, 0, :]

        return z_pred, z_t

    def update_ema(self):
        if self.use_ema:
            with torch.no_grad():
                for param_q, param_k in zip(
                    self.context_encoder.parameters(), self.target_encoder.parameters()
                ):
                    param_k.data = param_k.data * self.ema_decay + param_q.data * (
                        1.0 - self.ema_decay
                    )
