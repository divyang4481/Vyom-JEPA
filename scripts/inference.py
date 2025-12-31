import torch
from PIL import Image
import sys
import os
import yaml
import torchvision.transforms as transforms

# sys.path.append(os.getcwd())

from vyom_jepa.models.quantum_vl_jepa import QuantumVLJEPA


def main():
    # 1. Setup
    print("Loading Config and Model...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = QuantumVLJEPA(
        vision_model_name=config["model"].get("vision_model", "vit_small_patch16_224"),
        text_model_name=config["model"].get("text_model", "all-MiniLM-L6-v2"),
        predictor_hidden_dim=config["model"].get("predictor_hidden", 512),
    ).to(device)

    # Load Weights if available
    run_name = config.get("run_name", "vl_jepa_v1")
    ckpt_dir = os.path.join("experiments", run_name, "checkpoints")
    latest_ckpt = None
    if os.path.exists(ckpt_dir):
        ckpts = os.listdir(ckpt_dir)
        if ckpts:
            ckpts.sort(
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )  # numeric sort
            latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
            print(f"Loading checkpoint: {latest_ckpt}")
            state_dict = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(state_dict)

    if latest_ckpt is None:
        print("Warning: No checkpoint found. Using random weights.")

    model.eval()

    # 2. Input Data (Simulate a test case)
    print("\n--- Inference Input ---")
    query_text = "Describe the image."
    print(f"Query: {query_text}")

    # Create a dummy image (or load one)
    # Try to load a real image from the downloaded set
    test_img_path = "data/flickr8k/Images/real_dog.jpg"
    if os.path.exists(test_img_path):
        print(f"Loading REAL image: {test_img_path}")
        img = Image.open(test_img_path).convert("RGB")
    else:
        print("Image: Dummy Red Image (224x224)")
        img = Image.new("RGB", (224, 224), color="red")

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 3. Forward Pass
    print("\n--- Running Model ---")
    with torch.no_grad():
        # returns dict with 'z_pred_re', 'z_pred_im', 'z_target' (if target provided)
        output = model(img_tensor, [query_text], compute_target=False)

        z_re = output["z_pred_re"]
        z_im = output["z_pred_im"]

        # Calculate Magnitude (Quantum Probability Amplitude / Occupancy)
        # |psi|^2 = re^2 + im^2
        magnitude = torch.sqrt(z_re**2 + z_im**2)

        print("\n--- Output Analysis ---")
        print(f"Complex Latent Shape: {z_re.shape} (Real) + {z_im.shape} (Imag)")
        print(f"Latent Dimension: {z_re.shape[1]}")

        print("\nLatent Statistics:")
        print(f"  Max Magnitude: {magnitude.max().item():.4f}")
        print(f"  Mean Magnitude: {magnitude.mean().item():.4f}")
        print(f"  Real Part Mean: {z_re.mean().item():.4f} (Goal: ~0)")
        print(f"  Imag Part Mean: {z_im.mean().item():.4f} (Goal: ~0)")

        # In a real application, we would compare this 'z_pred'
        # to a database of text embeddings to find the best match (Retrieval/Generation)
        # For example, "A red image" embedding.

        print(
            "\nSuccess! The model successfully mapped (Image + Query) -> Complex Latent Space."
        )
        
        # --- Retrieval Test ---
        print("\n--- Retrieval Test from Real Data ---")
        candidates = []
        cap_file = "data/flickr8k/captions.txt"
        if os.path.exists(cap_file):
            with open(cap_file, 'r') as f:
                lines = f.readlines()[1:] # skip header
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        candidates.append(",".join(parts[1:]))
        
        # Fallbacks if file empty/missing
        if not candidates:
            candidates = ["A red cat", "A blue dog", "A synthetic image"]
            
        candidates = list(set(candidates))
        print(f"Candidates: {len(candidates)}")
        
        # Encode Candidates
        print("Encoding candidates... (This might take a moment)")
        cand_embs = model.encode_text(candidates, device)
        
        # Compute Fidelity
        # Normalize Pred
        norm = torch.norm(torch.stack([z_re, z_im], dim=0), dim=0).norm(dim=1, keepdim=True) + 1e-8
        z_re_norm = z_re / norm
        z_im_norm = z_im / norm
        
        # Normalize Cands
        cand_embs_norm = cand_embs / (cand_embs.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Dot Product
        dot_re = torch.mm(z_re_norm, cand_embs_norm.T)
        dot_im = torch.mm(z_im_norm, cand_embs_norm.T)
        fidelity = dot_re**2 + dot_im**2
        
        # Top 5
        probs = fidelity[0]
        top_k = min(5, len(candidates))
        vals, indices = torch.topk(probs, top_k)
        
        print(f"\nTop {top_k} Matches:")
        print("-" * 50)
        for i in range(top_k):
            idx = indices[i].item()
            score = vals[i].item()
            print(f"{score:.4f} | {candidates[idx]}")



if __name__ == "__main__":
    main()
