# 6GB VRAM Training Strategy

Training a modern Vision-Language model on a **6GB Laptop GPU** is challenging but possible with strict optimizations.

## 1. The Strategy: "Freeze & adapter"

We cannot finetune the huge encoders (ViT + BERT). We must keep them **Frozen**.
We only train the small **Predictor MLP**.

| Component                      | Size (MB) | State             | VRAM Usage                      |
| :----------------------------- | :-------- | :---------------- | :------------------------------ |
| **Vision Encoder** (ViT-Small) | ~88 MB    | Frozen (No Grads) | ~100 MB                         |
| **Text Encoder** (MiniLM)      | ~80 MB    | Frozen (No Grads) | ~100 MB                         |
| **Predictor** (MLP)            | ~50 MB    | **Trainable**     | ~200 MB (Weights + Grads + Opt) |
| **Activations** (Batch Size 8) | -         | -                 | ~2 GB                           |
| **Overhead** (PyTorch/CUDA)    | -         | -                 | ~1-2 GB                         |
| **Total**                      |           |                   | **~4 GB** (Safe Zone)           |

## 2. Techniques Implemented

### A. Mixed Precision (AMP)

We use `torch.cuda.amp` (Automatic Mixed Precision).

- **Effect**: Halves memory for activations (FP16 instead of FP32).
- **Status**: ✅ Already enabled in `train_vl.py`.

### B. Gradient Accumulation

Instead of a large batch (e.g., 128) which crashes memory, we use a **Micro-Batch** of 8.
We accumulate gradients over 16 steps to simulate a large batch of $8 \times 16 = 128$.

- **Effect**: Fits in VRAM while maintaining training stability.
- **Status**: ✅ Configurable in `config.yaml`.

### C. "Offline" Pre-Encoding (The Ultimate Hack)

If we run out of memory, we can pre-compute all image and text embeddings and save them to disk (using CPU or GPU inference mode).
Then, we train the Predictor using only these vectors.

- **VRAM Usage**: **< 1GB**.
- **Speed**: Extremely fast training.
- **Tradeoff**: Cannot use data augmentation on images easily.

## 3. Configuration for 6GB

Use these settings in `config.yaml`:

```yaml
model:
  vision_model: "vit_small_patch16_224" # Smallest ViT
  text_model: "all-MiniLM-L6-v2" # Smallest BERT
  predictor_hidden: 512 #keep it moderate

data:
  batch_size: 8 # KEY: Keep this small (<16)

training:
  mixed_precision: true # Crucial
  grad_accum_steps: 16 # Simulates batch size 128
```
