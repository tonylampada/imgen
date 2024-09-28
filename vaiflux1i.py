import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
import time
import random
import base64
from io import BytesIO
from PIL import Image

# Load the model and measure the time
start_time = time.time()
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
print(f"Model loading time: {time.time() - start_time:.2f} seconds")

prompt = "Red shirt"
seed = random.randint(0, 2**32 - 1)
generator = torch.Generator("cpu").manual_seed(seed)

init_image = Image.open("tonyroboflow.png").convert("RGB")
# Generate the image and measure the time
start_time = time.time()
image = pipe(
    prompt,
    image=init_image,
    strength=0.55,  # Adjust this value between 0 and 1 to control how much to change the initial image
    output_type="pil",
    num_inference_steps=50,
    generator=generator
).images[0]
print(f"Image generation time: {time.time() - start_time:.2f} seconds")

# Save the image to a file
image.save("flux2.png", format="PNG")
print("Image saved as flux.png")
