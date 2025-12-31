import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os
import sys
import yaml

# Add src to path to allow imports
# sys.path.append(os.path.join(os.getcwd(), 'src'))

from vyom_jepa.models.quantum_vl_jepa import QuantumVLJEPA

# Page Config
st.set_page_config(
    page_title="Vyom-JEPA Quantum Eye",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #00eeff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = QuantumVLJEPA(
        vision_model_name=config['model'].get('vision_model', 'vit_small_patch16_224'),
        text_model_name=config['model'].get('text_model', 'all-MiniLM-L6-v2'),
        predictor_hidden_dim=config['model'].get('predictor_hidden', 512)
    ).to(device)

    # Load Checkpoint
    run_name = config.get("run_name", "vl_jepa_v1")
    ckpt_dir = os.path.join("experiments", run_name, "checkpoints")
    
    status_text = "Initialized with random weights (No checkpoint found)"
    if os.path.exists(ckpt_dir):
        ckpts = os.listdir(ckpt_dir)
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
            state_dict = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(state_dict)
            status_text = f"Loaded Checkpoint: {latest_ckpt}"
            
    model.eval()
    return model, device, status_text

# --- Main App ---

st.title("👁️ Vyom-JEPA: Quantum Visual Cortex")
st.markdown("*Real-time Latent State Visualization & Semantic Retrieval*")

# Load Model
with st.spinner("Initializing Quantum Core..."):
    model, device, status = load_model()

st.sidebar.success(status)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration")

# Semantic Setup
query = st.sidebar.text_input("Query Context", value="Describe the object in the image.")
candidates_raw = st.sidebar.text_area(
    "Candidate Answers (Comma Separated)", 
    value="A person's face, A smartphone, A water bottle, An empty room, A cat, A dog"
)
candidates = [c.strip() for c in candidates_raw.split(",") if c.strip()]

# Camera
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Visual Input")
    img_file_buffer = st.camera_input("Capture State")

if img_file_buffer is not None:
    # Process Image
    img = Image.open(img_file_buffer).convert('RGB')
    
    # Transform
    import torchvision.transforms as transforms
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = img_transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor, [query], compute_target=False)
        z_re = output['z_pred_re']
        z_im = output['z_pred_im']
        
        # 1. State Visualization
        magnitude = torch.sqrt(z_re**2 + z_im**2).cpu().numpy().flatten()
        phase = torch.atan2(z_im, z_re).cpu().numpy().flatten()
        
        with col2:
            st.subheader("🧠 Quantum Latent State")
            
            # Heatmap of complex vector state
            fig_state = go.Figure(data=go.Heatmap(
                z=[magnitude],
                colorscale='Viridis',
                showscale=False
            ))
            fig_state.update_layout(
                title="State Occupancy (Magnitude |ψ|)", 
                height=150, 
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_state, use_container_width=True)
            
            st.metric("Mean Coherence", f"{magnitude.mean():.4f}")

        # 2. Semantic Retrieval (Classification)
        if candidates:
            # Encode candidates
            cand_embs = model.encode_text(candidates, device)
            
            # Fidelity
            norm = torch.norm(torch.stack([z_re, z_im], dim=0), dim=0).norm(dim=1, keepdim=True) + 1e-8
            z_re_norm = z_re / norm
            z_im_norm = z_im / norm
            cand_embs_norm = cand_embs / (cand_embs.norm(dim=-1, keepdim=True) + 1e-8)
            
            dot_re = torch.mm(z_re_norm, cand_embs_norm.T)
            dot_im = torch.mm(z_im_norm, cand_embs_norm.T)
            fidelity = (dot_re**2 + dot_im**2).cpu().numpy().flatten()
            
            # Plot
            st.markdown("### 🔍 Semantic Resonance (Probability)")
            
            # Normalize for visualization
            # fidelity = fidelity / (fidelity.sum() + 1e-8)
            
            fig_bar = go.Figure(go.Bar(
                x=fidelity,
                y=candidates,
                orientation='h',
                marker=dict(
                    color=fidelity,
                    colorscale='Plasma'
                )
            ))
            fig_bar.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                xaxis_title="Fidelity |<ψ|y>|²"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            best_idx = np.argmax(fidelity)
            st.success(f"**Best Match:** {candidates[best_idx]}")

else:
    with col2:
        st.info("Waiting for visual input...")
