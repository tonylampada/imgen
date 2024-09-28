import sys
import torch
from diffusers import FluxImg2ImgPipeline
import random
import base64
from io import BytesIO
from PIL import Image

def generate_img2img(prompt, input_image_path, strength):
    # Load the model
    model_id = "black-forest-labs/FLUX.1-dev"
    pipe = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    try:
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        # Load the input image
        init_image = Image.open(input_image_path).convert("RGB")

        image = pipe(
            prompt,
            image=init_image,
            strength=float(strength),
            output_type="pil",
            num_inference_steps=50,
            generator=generator
        ).images[0].resize(init_image.size)

        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode()
        
        print(img_str)  # This will be captured by the subprocess
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python imgen_img2img.py <prompt> <input_image_path> <strength>", file=sys.stderr)
        sys.exit(1)
    
    prompt = sys.argv[1]
    input_image_path = sys.argv[2]
    strength = sys.argv[3]
    generate_img2img(prompt, input_image_path, strength)
