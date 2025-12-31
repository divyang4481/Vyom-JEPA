import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import yaml
import sys
import os
import time
from tqdm import tqdm
import torchvision.transforms as transforms

# Ensure imports work
sys.path.append(os.getcwd())

from vyom_jepa.models.quantum_vl_jepa import QuantumVLJEPA
from vyom_jepa.losses.quantum_losses import FidelityLoss, SIGRegCLoss
from vyom_jepa.data.vl_dataset import VLDataset


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1. Setup
    config = load_config()

    # Force CUDA
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available but required for 6GB VRAM Strategy! Check valid PyTorch installation."
        )

    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    # Memory check
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {vram:.2f} GB")
    if vram < 5.0:
        print("Warning: VRAM < 5GB. Ensure batch_size is small!")

    run_name = config.get("run_name", "vl_jepa_default")
    output_dir = os.path.join("experiments", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Data
    data_cfg = config["data"]
    ds_type = data_cfg.get("dataset_type", "vl_synthetic")

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if ds_type == "vl_synthetic" or ds_type == "flickr8k":
        dataset = VLDataset(
            root_dir="data/flickr8k",  # Placeholder
            dataset_type=ds_type,
            transform=transform,
        )
    else:
        raise ValueError("Use synthetic or flickr8k for this script")

    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
    )

    # 3. Model
    model_cfg = config["model"]
    print("Loading QuantumVLJEPA...")
    model = QuantumVLJEPA(
        vision_model_name=model_cfg.get("vision_model", "vit_small_patch16_224"),
        text_model_name=model_cfg.get("text_model", "all-MiniLM-L6-v2"),
        predictor_hidden_dim=model_cfg.get("predictor_hidden", 512),
        freeze_encoders=True,  # Default True for VRAM safety
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable Parameters (Predictor): {model_params:.2f}M")

    # Save config snapshot
    with open(os.path.join(output_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)

    # 4. Losses & Optimizer
    fidelity_loss_fn = FidelityLoss()
    sigreg_loss_fn = SIGRegCLoss(alpha=0.1)  # alpha form config?

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["training"]["lr"]),
    )
    scaler = (
        torch.amp.GradScaler("cuda") if config["training"]["mixed_precision"] else None
    )
    grad_accum_steps = config["training"].get("grad_accum_steps", 1)

    # 5. Loop
    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        start_time = time.time()
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{epochs}",
        )

        for i, batch in pbar:
            images = batch["image"].to(device)
            queries = batch["query"]
            targets = batch["target"]

            with torch.amp.autocast(
                "cuda", enabled=config["training"]["mixed_precision"]
            ):
                # Forward
                out = model(images, queries, targets)

                z_re = out["z_pred_re"]
                z_im = out["z_pred_im"]
                target_real = out["z_target"]

                # Losses
                loss_fid = fidelity_loss_fn(z_re, z_im, target_real)
                loss_reg = sigreg_loss_fn(z_re, z_im)

                loss = loss_fid + loss_reg
                loss = loss / grad_accum_steps

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item() * grad_accum_steps
            total_loss += current_loss
            pbar.set_postfix(
                {"loss": f"{current_loss:.4f}", "fid": f"{loss_fid.item():.4f}"}
            )

        avg_loss = total_loss / len(dataloader)
        dt = time.time() - start_time
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {dt:.2f}s")

        # Logs
        log_path = os.path.join(output_dir, "training_log.csv")
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("Epoch,Loss,Time\n")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss},{dt}\n")

        # Checkpoints
        if (epoch + 1) % 5 == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pt")
            )


if __name__ == "__main__":
    main()
