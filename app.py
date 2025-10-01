import streamlit as st
import torch
from siren import Siren
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    # Must match training config in sirenMRI_2D.py
    model = Siren2D(
        dim_in=2,          # 2D coords
        dim_hidden=256,    # hidden size (default in repo)
        dim_out=1,         # grayscale output
        num_layers=5,      # depth
        w0=30.,
        w0_initial=30.
    )
    model.load_state_dict(torch.load("best_model_slice.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 SIREN MRI Compression Demo (NIfTI)")
st.write("Upload a **NIfTI MRI volume (.nii or .nii.gz)** and reconstruct a slice using the trained SIREN2D model.")

uploaded_file = st.file_uploader("Upload NIfTI file", type=["nii", "nii.gz"])

if uploaded_file is not None:
    # Load NIfTI file
    img = nib.load(uploaded_file)
    data = img.get_fdata()

    st.write(f"Shape of MRI volume: {data.shape}")
    slice_idx = st.slider("Choose slice index", 0, data.shape[2]-1, data.shape[2]//2)

    # Extract chosen slice
    slice_img = data[:, :, slice_idx]
    slice_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())

    st.write("### Original MRI Slice")
    st.image(slice_norm, clamp=True, use_column_width=True)

    # Prepare coordinates in [-1, 1]
    h, w = slice_norm.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing="ij"
    )
    coords = torch.stack([x.flatten(), y.flatten()], dim=-1)

    # Run model
    with torch.no_grad():
        preds = model(coords).cpu().numpy().reshape(h, w)

    preds_norm = (preds - preds.min()) / (preds.max() - preds.min())

    st.write("### Reconstructed Slice (SIREN)")
    st.image(preds_norm, clamp=True, use_column_width=True)

    # Option to save reconstructed slice
    st.download_button(
        "Download Reconstructed Slice (.npy)",
        data=preds_norm.astype(np.float32).tobytes(),
        file_name="reconstructed_slice.npy",
        mime="application/octet-stream"
    )
