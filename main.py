# Install necessary libraries
!pip install -q diffusers transformers accelerate torch torchvision

# Import necessary libraries
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import huggingface_hub

# Set your Hugging Face token
# Make sure to replace 'YOUR_HUGGINGFACE_TOKEN' with your actual token
huggingface_hub.login(token='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') 

# Load the Stable Diffusion model
# This is a different model ID 
model_id = "stabilityai/stable-diffusion-2-1"  
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate an image from a text prompt
def generate_image(prompt):
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

# Provide your text prompt
prompt = "A fantasy landscape with mountains, waterfalls, and a magical castle"
generated_image = generate_image(prompt)

# Display the generated image
generated_image.show()
