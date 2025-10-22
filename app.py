import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from torchvision import transforms
from PIL import Image
import numpy as np
import nibabel as nib
import io
import os

# ----------------------------------
# Load your trained model
# ----------------------------------
def load_model():
    model_path = hf_hub_download(
        repo_id="medimaging/physics-informed-sirenMRI-model",  # your HF model repo
        filename="sirenMRI_full_model.pt"  # ensure this matches your .pt file name
    )
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

model = load_model()

# ----------------------------------
# Image Preprocessing / Postprocessing
# ----------------------------------
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust if needed
    transforms.ToTensor(),
])
postprocess = transforms.ToPILImage()

# ----------------------------------
# Utility functions
# ----------------------------------
def psnr(img1, img2):
    """Compute PSNR between two tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def estimate_compression_ratio(original_img, model_params):
    """Estimate compression ratio (rough)."""
    img_bytes = original_img.size[0] * original_img.size[1] * 3
    model_bytes = model_params * 4  # 4 bytes per 32-bit float
    return round(img_bytes / model_bytes, 2)

def extract_middle_slice(nii_path):
    """Load NIfTI image and extract the middle slice as a PIL Image."""
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    mid_slice = data.shape[2] // 2  # take middle z-slice
    slice_img = data[:, :, mid_slice]
    slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-8)
    slice_img = (slice_img * 255).astype(np.uint8)
    return Image.fromarray(slice_img)

# ----------------------------------
# Core compression + reconstruction logic
# ----------------------------------
def compress_and_reconstruct(file):
    """
    Handles both MRI (.nii/.nii.gz) and image (.png/.jpg) inputs.
    Returns reconstructed image and metrics.
    """
    # Determine input type
    filename = getattr(file, "name", "")
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        image = extract_middle_slice(filename)
    else:
        image = Image.open(file).convert("RGB")

    original = preprocess(image).unsqueeze(0)

    # --- Compression simulation (inference) ---
    with torch.no_grad():
        reconstructed = model(original)

    reconstructed = reconstructed.squeeze(0).clamp(0, 1)

    # Compute metrics
    score = psnr(original, reconstructed).item()
    ratio = estimate_compression_ratio(image, sum(p.numel() for p in model.parameters()))

    # Convert to images
    orig_img = postprocess(original.squeeze(0))
    recon_img = postprocess(reconstructed)

    # Combine into side-by-side view
    combined = Image.new("RGB", (orig_img.width * 2, orig_img.height))
    combined.paste(orig_img, (0, 0))
    combined.paste(recon_img, (orig_img.width, 0))

    return combined, f"PSNR: {score:.2f} dB | Compression Ratio: {ratio}:1"

# ----------------------------------
# Gradio UI
# ----------------------------------
title = "🧠 Physics-Informed SIREN MRI Compression"
description = """
Upload a **MRI file (.nii / .nii.gz)** or **image (PNG/JPG)**  
to visualize compression and reconstruction using the **Physics-Informed SIREN** model.

The demo shows:
- Side-by-side comparison *(Original → Reconstructed)*
- PSNR & Compression Ratio metrics

**Author:** medimaging  
**Framework:** PyTorch + Gradio  
"""

interface = gr.Interface(
    fn=compress_and_reconstruct,
    inputs=gr.File(label="Upload MRI or Image File (.nii, .nii.gz, .png, .jpg)"),
    outputs=[
        gr.Image(label="Original (Left) → Reconstructed (Right)"),
        gr.Textbox(label="Quality Metrics", interactive=False)
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()
