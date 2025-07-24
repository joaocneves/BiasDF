import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import argparse


def generate_images(input_csv, output_dir="images_diff"):
    # Load the CSV file
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"❌ Input file not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    if "description" not in df.columns:
        raise ValueError("❌ CSV must contain a 'description' column.")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Stable Diffusion model
    print("🚀 Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate 4 images per prompt
    print(f"📦 Starting generation for {len(df)} prompts...")
    for idx, row in df.iterrows():
        prompt = str(row['description'])
        print(f"🔹 [{idx}] Prompt: {prompt}")
        for i in range(1, 5):
            image = pipe(prompt).images[0]
            filename = f"img_diff_{idx}_{i}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"    ✅ Saved: {filepath}")

    print(f"\n✅ All images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion from a CSV file of prompts.")
    parser.add_argument("--input", required=True, help="Path to the CSV file containing 'description' column.")
    parser.add_argument("--output", default="images_diff", help="Directory to save generated images (default: images_diff)")

    args = parser.parse_args()
    generate_images(args.input, args.output)
