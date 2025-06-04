import sys
import os
import argparse
from PIL import Image

# Add the path to the thirdparty/SeeSR directory to the Python path
sys.path.append(os.path.abspath("./thirdparty/SeeSR"))

import torch
from torchvision import transforms
from ram.models.ram_lora import ram
from ram import inference_ram as inference

def load_ram_model(ram_model_path: str, dape_model_path: str):
    """
    Load the RAM model with the given paths.

    Args:
        ram_model_path (str): Path to the pretrained RAM model.
        dape_model_path (str): Path to the pretrained DAPE model.

    Returns:
        torch.nn.Module: Loaded RAM model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the RAM model
    tag_model = ram(pretrained=ram_model_path, pretrained_condition=dape_model_path, image_size=384, vit="swin_l")
    tag_model.eval()
    return tag_model.to(device)

def generate_caption(image_path: str, tag_model) -> str:
    """
    Generate a caption for a degraded image using the RAM model.

    Args:
        image_path (str): Path to the degraded input image.
        tag_model (torch.nn.Module): Preloaded RAM model.

    Returns:
        str: Generated caption for the image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define image transformations
    tensor_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = tensor_transforms(image).unsqueeze(0).to(device)
    image_tensor = ram_transforms(image_tensor)

    # Generate caption using the RAM model
    caption = inference(image_tensor, tag_model)

    return caption[0]

def process_images_in_directory(input_dir: str, output_file: str, tag_model):
    """
    Process all images in a directory, generate captions using the RAM model,
    and save the captions to a file.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_file (str): Path to the file where captions will be saved.
        tag_model (torch.nn.Module): Preloaded RAM model.
    """
    # Open the output file for writing captions
    with open(output_file, "w") as f:
        # Iterate through all files in the input directory
        for filename in os.listdir(input_dir):
            # Construct the full path to the image file
            image_path = os.path.join(input_dir, filename)

            # Check if the file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Generate a caption for the image
                    caption = generate_caption(image_path, tag_model)
                    print(f"Generated caption for {filename}: {caption}")
                    # Write the caption to the output file
                    f.write(f"{filename}: {caption}\n")

                    print(f"Processed {filename}: {caption}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images using RAM and DAPE models.")
    parser.add_argument("--input_dir", type=str, default="data/val", help="Path to the directory containing input images.")
    parser.add_argument("--output_file", type=str, default="data/val_captions.txt", help="Path to the file where captions will be saved.")
    parser.add_argument("--ram_model", type=str, default="/home/erbachj/scratch2/projects/var_post_samp/thirdparty/SeeSR/preset/model/ram_swin_large_14m.pth", help="Path to the pretrained RAM model.")
    parser.add_argument("--dape_model", type=str, default="/home/erbachj/scratch2/projects/var_post_samp/thirdparty/SeeSR/preset/model/DAPE.pth", help="Path to the pretrained DAPE model.")

    args = parser.parse_args()

    # Load the RAM model once
    tag_model = load_ram_model(args.ram_model, args.dape_model)

    # Process images in the directory
    process_images_in_directory(args.input_dir, args.output_file, tag_model)