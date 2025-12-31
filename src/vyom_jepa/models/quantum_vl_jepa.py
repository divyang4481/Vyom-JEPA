import torch
import torch.nn as nn
import timm
from sentence_transformers import SentenceTransformer


class ComplexPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        # Predictor takes [Vision_Emb, Query_Emb] -> Complex Latent (Real, Imag)
        # Input dim is Vision_Dim + Query_Dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            # Output 2 * output_dim for Real/Imag
            nn.Linear(hidden_dim, 2 * output_dim),
        )

    def forward(self, x):
        # x: [B, Input_Dim]
        out = self.net(x)
        # Split into Real and Imag
        re, im = out.chunk(2, dim=-1)
        return re, im


class QuantumVLJEPA(nn.Module):
    def __init__(
        self,
        vision_model_name="vit_small_patch16_224",
        text_model_name="all-MiniLM-L6-v2",
        predictor_hidden_dim=512,
        freeze_encoders=True,
    ):
        super().__init__()

        # 1. Vision Encoder (X-Encoder)
        # Using timm
        self.vision_encoder = timm.create_model(
            vision_model_name, pretrained=True, num_classes=0
        )
        self.vision_dim = self.vision_encoder.num_features

        # 2. Text Encoder (Y-Encoder & Query Encoder)
        # Using Sentence Transformers (HuggingFace)
        # We wrap it to be torch compatible
        self.text_model_name = text_model_name
        # Placeholder, will load in actual script to avoid obscure pickling/weights issues if possible,
        # or load here. SentenceTransformer is heavy.
        # Let's assume we pass pre-computed embeddings OR we allow finetuning.
        # For VRAM efficiency: We might want to compute text embeddings *offline* or
        # keep the model on CPU if it's large.
        # But 'all-MiniLM-L6-v2' is tiny (80MB). We can load it.
        # Problem: SentenceTransformer is not a standard nn.Module sometimes.
        # Better to use HF Transformers directly if trainable.
        # But prompt says "freeze by default".

        # We will use a small HF model.
        from transformers import AutoModel, AutoTokenizer

        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_dim = 384  # Known for MiniLM

        if freeze_encoders:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # 3. Predictor
        # Vision + Query -> Target (Complex)
        # Query is Text. Target is Text.
        # Input to predictor = Concat(Vision, Query)
        input_dim = self.vision_dim + self.text_dim
        self.predictor = ComplexPredictor(
            input_dim, predictor_hidden_dim, self.text_dim
        )

    def encode_text(self, text_list, device):
        # Allow batch encoding
        inputs = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():  # Assuming frozen mostly
            outputs = self.text_encoder(**inputs)
        # Mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return embeddings

    def forward(self, images, queries, targets=None, compute_target=True):
        """
        Args:
            images: [B, C, H, W]
            queries: List of strings or [B, D] embeddings
            targets: List of strings (optional, for training)
        """
        device = images.device

        # 1. Vision Embedding
        # [B, V_Dim]
        vision_emb = self.vision_encoder(images)

        # 2. Query Embedding
        if isinstance(queries[0], str):
            query_emb = self.encode_text(queries, device)
        else:
            query_emb = queries

        # 3. Predict (Vision + Query) -> Complex Latent
        # Concatenate
        combined = torch.cat([vision_emb, query_emb], dim=-1)
        z_re, z_im = self.predictor(combined)

        # 4. Target Embedding (Real)
        target_emb = None
        if compute_target and targets is not None:
            if isinstance(targets[0], str):
                target_emb = self.encode_text(targets, device)
            else:
                target_emb = targets

        return {
            "z_pred_re": z_re,
            "z_pred_im": z_im,
            "z_target": target_emb,  # Real valued target
        }
