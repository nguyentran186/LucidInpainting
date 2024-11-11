import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def load_mask(mask_path):
    # Assuming the mask is a grayscale image, convert to binary mask (0 or 255)
    mask = Image.open(mask_path).convert("L")
    return mask

def run_inpainting(pipe, image, mask, prompt=""):
    # Resize both image and mask to the model's expected input resolution (e.g., 512x512)
    image = image.resize((512, 512))
    mask = mask.resize((512, 512))
    
    # Run the inpainting pipeline
    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    
    return result

if __name__ == "__main__":
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    )
    pipe.to(device)

    # Load your input image and mask
    image_path = "/root/LucidInpainting/data/statue/images/IMG_2707.jpg"
    mask_path = "/root/LucidInpainting/data/statue/seg/IMG_2707.jpg"
    image = load_image(image_path)
    mask = load_mask(mask_path)

    # Set a prompt for inpainting
    prompt = "A monkey"

    # Run inpainting
    inpainted_image = run_inpainting(pipe, image, mask, prompt)
    
    # Save the result
    inpainted_image.save("output_inpainted.png")
