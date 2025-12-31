import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import yaml
import sys
import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")


# Ensure we can import from local modules if running as script
# If running 'python src/train.py' from root, sys.path[0] is .../src
# so 'import vyom_jepa' works.

from vyom_jepa.models.vyom_jepa import VyomJEPA
from vyom_jepa.losses.sigreg import SIGRegLoss
from vyom_jepa.data.fock_dataset import FockDynamicsDataset
from vyom_jepa.data.spin_chain_dataset import SpinChainDataset


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1. Setup
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    data_cfg = config["data"]
    ds_type = data_cfg.get("dataset_type", "fock")

    if ds_type == "spin_chain":
        print("Loading Spin Chain Dataset...")
        dataset = SpinChainDataset(
            size=data_cfg["size"],
            num_spins=data_cfg.get("num_spins", 16),
            J=1.0,
        )
        # Input dim is 3 * N (positions x,y,z)
        input_dim = 3 * data_cfg.get("num_spins", 16)

    else:
        print("Loading Fock Dataset...")
        dataset = FockDynamicsDataset(
            size=data_cfg["size"],
            num_modes=16,
            max_occupation=5,
        )
        input_dim = 16

    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
    )

    # 3. Model
    model_cfg = config["model"]
    model = VyomJEPA(
        input_dim=input_dim,  # Dynamic input dim
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        predictor_layers=model_cfg["predictor_layers"],
        use_ema=model_cfg["use_ema"],
        ema_decay=model_cfg["ema_decay"],
    ).to(device)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Compile
    if config["training"].get("compile", False):
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Compilation failed: {e}. Proceeding without compilation.")

    # 4. Losses & Optimizer
    # SIGReg Loss
    sigreg_loss_fn = SIGRegLoss(
        feature_dim=model_cfg["d_model"], num_projections=16
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))
    scaler = (
        torch.amp.GradScaler("cuda") if config["training"]["mixed_precision"] else None
    )

    alpha = config["training"]["alpha_sigreg"]
    grad_accum_steps = config["training"].get("grad_accum_steps", 1)

    # 5. Training Loop
    epochs = config["training"]["epochs"]
    from tqdm import tqdm

    # Setup run directory
    run_name = config.get("run_name", "default_run")
    # Add timestamp to run_name to avoid overwrites if not manually changed
    # run_name = f"{run_name}_{int(time.time())}"
    # Or keep it deterministic as per user request to group by version.

    output_dir = os.path.join("experiments", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(output_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)

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
            # Move data
            x_context = batch["x_context"].to(device)
            x_target = batch["x_target"].to(device)
            dt = batch["dt"].to(device)  # [B, 1]

            # AMP Context
            with torch.amp.autocast(
                "cuda", enabled=config["training"]["mixed_precision"]
            ):
                # Forward
                z_pred, z_target = model(x_context, x_target, dt)

                # Losses
                # 1. Prediction Loss (MSE between predicted latent and target latent)
                pred_loss = nn.functional.mse_loss(z_pred, z_target)

                # 2. Regularization (SIGReg) on target representation
                # Applying to z_target ensures the targets (which guide training) are well-distributed.
                # Often applied to context as well. We'll apply to both or just target.
                # LeJEPA implies regularization on the representations.
                reg_loss = sigreg_loss_fn(z_target)

                loss = pred_loss + alpha * reg_loss
                loss = loss / grad_accum_steps

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step
            if (i + 1) % grad_accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                # Update EMA Target Encoder
                if hasattr(model, "update_ema") and model.use_ema:
                    model.update_ema()
                elif hasattr(model, "_orig_mod") and hasattr(
                    model._orig_mod, "update_ema"
                ):
                    # Handle compiled model wrapping
                    model._orig_mod.update_ema()

            current_loss = loss.item() * grad_accum_steps
            total_loss += current_loss
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_loss = total_loss / len(dataloader)
        dt_epoch = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} Completed | Avg Loss: {avg_loss:.4f} | Time: {dt_epoch:.2f}s"
        )

        # Save logs
        log_path = os.path.join(output_dir, "training_log.csv")
        # Write header if new file
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("Epoch,Loss,Time\n")

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss},{dt_epoch}\n")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pt")
            )


if __name__ == "__main__":
    main()
