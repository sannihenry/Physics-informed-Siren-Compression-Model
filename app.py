import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

# 🔹 Download model from Hugging Face
model_path = hf_hub_download(repo_id="medimaging/physics-informed-sirenMRI-model", filename="sirenMRI_full_model.pt")
model = torch.load(model_path, map_location="cpu")
model.eval()

def compress_mri(image):
    # Example placeholder — replace this with your real inference logic
    image = image.convert("L")  # grayscale MRI slice
    input_tensor = torch.tensor(np.array(image) / 255.0).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = input_tensor  # <-- replace with model(image) once ready
    
    output_img = Image.fromarray((output.squeeze().numpy() * 255).astype(np.uint8))
    return output_img

demo = gr.Interface(
    fn=compress_mri,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=gr.Image(type="pil", label="Compressed Output"),
    title="SIREN MRI Compression Model",
    description="Physics-informed medical image compression using Physics-Informed-SIREN."
)

if __name__ == "__main__":
    demo.launch()
