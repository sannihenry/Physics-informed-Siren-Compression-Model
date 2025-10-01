import os
import torch
import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Import your SIREN model definition
from siren import SirenNet   # adjust if your class name is different

# ----------------------
# Load model
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ⚠️ Update input/output sizes and hidden dims to match your training
model = SirenNet(
    in_features=3,          # e.g. x,y,z coords or 2 for 2D
    hidden_features=256,
    hidden_layers=3,
    out_features=1
)
model.load_state_dict(torch.load("best_model_slice_95.pt", map_location=device))
model.to(device)
model.eval()

# ----------------------
# Streamlit UI
# ----------------------
st.title("🧠 SirenMRI Compression Model")
st.write("Upload an MRI volume (.nii or .nii.gz) to reconstruct using the trained SIREN model.")

uploaded_file = st.file_uploader("Choose a NIfTI file", type=["nii", "nii.gz"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file = "uploaded.nii.gz"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load MRI data
    img = nib.load(temp_file)
    data = img.get_fdata()
    st.write(f"Input volume shape: {data.shape}")

    # Pick a middle slice for demo
    slice_idx = data.shape[-1] // 2
    input_slice = data[:, :, slice_idx]

    # Normalize input if needed
    input_slice = (input_slice - np.min(input_slice)) / (np.max(input_slice) - np.min(input_slice) + 1e-8)

    # Prepare coordinates for SIREN (example for 2D)
    H, W = input_slice.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    coords_torch = torch.tensor(coords, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(coords_torch).cpu().numpy().reshape(H, W)

    # Show original vs reconstructed
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(input_slice, cmap="gray")
    axes[0].set_title("Original Slice")
    axes[0].axis("off")
    axes[1].imshow(preds, cmap="gray")
    axes[1].set_title("Reconstructed Slice")
    axes[1].axis("off")

    st.pyplot(fig)
