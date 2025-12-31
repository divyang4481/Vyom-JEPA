# Vyom-JEPA (व्योम-JEPA)

**Quantum-Latent Predictive Architecture for Fock-Space Representations.**

Vyom-JEPA is an "Antigenerative" AI framework. Instead of predicting pixels or tokens, it predicts the latent evolution of quantum states. It bridges Meta's **Joint Embedding Predictive Architecture (JEPA)** with **Quantum Field Theory** principles.

## 🚀 Vision

The goal of Vyom-JEPA is to model the "Vyom" (Infinite Latent Space) where quantum occupancy states evolve. By using **LeJEPA** principles, we eliminate the need for computationally expensive generative decoders, making quantum simulation possible on consumer-grade hardware (6GB VRAM).

## 🛠 Key Features

- **Non-Generative:** Predicts embeddings, not raw data.
- **SIGReg Regularization:** Prevents representation collapse using Isotropic Gaussian constraints.
- **Indra-Ready:** Designed to integrate with the IndraQuantum Transformer backbone.
- **Physics-Informed:** Native support for Fock space occupancy vectors.

## 📦 Installation

Prerequisites:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed.
- A CUDA-capable GPU is recommended.

1. Clone the repository:

```bash
git clone https://github.com/divyang4481/Vyom-JEPA.git
cd Vyom-JEPA
```

2. Activate the environment:

```bash
conda activate venvindraquantum
```

3. Install the package in editable mode:

````bash
```bash
pip install -e .
pip install timm transformers sentence-transformers bitsandbytes
````

## 📂 Dataset Setup (Flickr8k)

To get real-world performance:

1.  **Download** Flickr8k from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).
2.  **Extract** to `data/flickr8k/`.
    - Ensure `captions.txt` is at `data/flickr8k/captions.txt`.
    - Ensure images are at `data/flickr8k/Images/*.jpg`.
3.  **Configure**: Set `dataset_type: "flickr8k"` in `config.yaml`.

## 🏁 Quick Start

### 1. Train the Model

Train the 150M parameter Quantum Predictor using your local GPU (6GB+ VRAM required):

```bash
python scripts/train.py
```

- Config: Modifiable in `config.yaml`
- Logs: Saved to `experiments/vl_jepa_v1/`

### 2. Run Inference

Test the trained model vs specific images and captions:

```bash
python scripts/inference.py
```

### 3. Interactive Webcam Demo

Launch the **Quantum Eye** real-time visualization:

```bash
streamlit run scripts/demo.py
```

This opens a web UI where you can see the model's "Quantum State" update live as you move objects in front of the camera.

## 📚 Documentation

- [Quantum-VL-JEPA Architecture](docs/QUANTUM_VL_JEPA.md)
- [Production Scaling Report](docs/PRODUCTION_SCALING.md)
- [6GB VRAM Optimization Strategy](docs/6GB_VRAM_STRATEGY.md)
- [Business Use Cases](docs/BUSINESS_USE_CASES.md)
- [Roadmap](docs/ROADMAP.md)
