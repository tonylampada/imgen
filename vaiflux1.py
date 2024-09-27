import torch
from diffusers import FluxPipeline
import time
import random
import base64
from io import BytesIO

# Load the model and measure the time
start_time = time.time()
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
print(f"Model loading time: {time.time() - start_time:.2f} seconds")

prompt = "A realistic phot of an alligator playing basketball"
seed = random.randint(0, 2**32 - 1)
generator = torch.Generator("cpu").manual_seed(seed)

# Generate the image and measure the time
start_time = time.time()
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=50,
    generator=generator
).images[0]
print(f"Image generation time: {time.time() - start_time:.2f} seconds")

# Save the image to a file
image.save("flux.png", format="PNG")
print("Image saved as flux.png")
