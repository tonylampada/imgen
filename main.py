import subprocess
import base64
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Body
from fastapi.responses import Response
import os
from pydantic import BaseModel

app = FastAPI()

class Img2ImgRequest(BaseModel):
    prompt: str
    image_base64: str
    strength: float = 0.55

@app.get("/generate")
async def generate_image(prompt: str = Query(..., description="The prompt for image generation")):
    try:
        # Call the AI processor as a subprocess
        result = subprocess.run(["python", "imgen.py", prompt], capture_output=True, text=True, check=True)
        
        # The AI processor will return the base64 encoded image
        img_str = result.stdout.strip()
        
        return Response(content=img_str, media_type="image/png;base64")
    except subprocess.CalledProcessError as e:
        print(f"AI processor error: {e.stderr}")
        raise HTTPException(status_code=500, detail="AI processing failed")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

@app.post("/generate_img2img")
async def generate_img2img(request: Img2ImgRequest = Body(...)):
    try:
        # Decode the base64 string to binary data
        image_data = base64.b64decode(request.image_base64)

        # Save the decoded image temporarily
        temp_image_path = "temp_input_image.png"
        with open(temp_image_path, "wb") as temp_file:
            temp_file.write(image_data)

        # Call the AI processor as a subprocess
        result = subprocess.run(
            ["python", "imgen_img2img.py", request.prompt, temp_image_path, str(request.strength)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Clean up the temporary file
        os.remove(temp_image_path)
        
        # The AI processor will return the base64 encoded image
        img_str = result.stdout.strip()
        
        return Response(content=img_str, media_type="image/png;base64")
    except subprocess.CalledProcessError as e:
        print(f"AI processor error: {e.stderr}")
        raise HTTPException(status_code=500, detail="AI processing failed")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
