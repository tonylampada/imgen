import sys
import torch
from diffusers import FluxPipeline
import random
import base64
from io import BytesIO

def generate_image(prompt):
    # Load the model
    model_id = "black-forest-labs/FLUX.1-dev"
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    try:
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=50,
            generator=generator
        ).images[0]

        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode()
        
        print(img_str)  # This will be captured by the subprocess
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ai_processor.py <prompt>", file=sys.stderr)
        sys.exit(1)
    
    prompt = sys.argv[1]
    generate_image(prompt)