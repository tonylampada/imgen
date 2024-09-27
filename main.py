import torch
from diffusers import FluxPipeline
import time
import random
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

# Load the model once when the application starts
model_id = "black-forest-labs/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

@app.get("/generate")
async def generate_image(prompt: str = Query(..., description="The prompt for image generation")):
    try:
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=4,
            generator=generator
        ).images[0]

        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode()
        print(img_str[:100])
        return Response(content=img_str, media_type="image/png;base64")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
