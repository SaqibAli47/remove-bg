from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
from fastapi.middleware.cors import CORSMiddleware
import requests
from pathlib import Path
import os
from scipy.ndimage import binary_erosion, binary_dilation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CACHE_DIR = "/tmp/models"
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "isnet-general-use.onnx")

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

def load_model():
    if not os.path.exists(MODEL_PATH):
        download_model()
    return ort.InferenceSession(MODEL_PATH)

def preprocess_image(img):
    # Resize and normalize the image
    img = img.convert('RGB')
    img = img.resize((320, 320))
    img = np.array(img) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return img.astype(np.float32)

def postprocess_mask(mask, threshold=0.5):
    mask = mask > threshold
    # Apply some morphological operations to clean up the mask
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=2)
    return (mask * 255).astype(np.uint8)

class ImageRequest(BaseModel):
    imageBase64: str

@app.post("/api/remove-bg")
async def remove_background(request: ImageRequest):
    try:
        # Initialize model
        model = load_model()
        
        # Decode base64 image
        image_data = base64.b64decode(request.imageBase64)
        input_image = Image.open(io.BytesIO(image_data))
        original_size = input_image.size
        
        # Preprocess
        img = preprocess_image(input_image)
        
        # Run inference
        mask = model.run(None, {'input': img})[0][0, 0]
        
        # Postprocess
        mask = postprocess_mask(mask)
        mask = Image.fromarray(mask).resize(original_size)
        
        # Apply mask to original image
        input_image = input_image.convert('RGBA')
        input_array = np.array(input_image)
        mask_array = np.array(mask)
        input_array[:, :, 3] = mask_array
        
        # Convert back to base64
        output_image = Image.fromarray(input_array)
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG", optimize=True)
        output_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"imageBase64": output_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing background: {str(e)}")

@app.get("/api/healthcheck")
async def read_root():
    model_exists = os.path.exists(MODEL_PATH)
    return {
        "status": "ok",
        "message": "Background Removal API is running",
        "model_ready": model_exists
    }
