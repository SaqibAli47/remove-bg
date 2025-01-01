from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import onnxruntime as ort
import cv2
import aiohttp
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model URL and path
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_PATH = "/tmp/u2netp.onnx"

async def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        async with aiohttp.ClientSession() as session:
            async with session.get(MODEL_URL) as response:
                with open(MODEL_PATH, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)

# Initialize the ONNX Runtime session
session = None

def get_session():
    global session
    if session is None and os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH)
    return session

def preprocess(img):
    img = img.convert('RGB')
    img = img.resize((320, 320))
    img = np.array(img)
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, ...]
    return img.astype(np.float32)

def postprocess(pred, size):
    pred = pred.squeeze()
    pred = cv2.resize(pred, size)
    pred = np.where(pred > 0.5, 255, 0)
    return Image.fromarray(pred.astype(np.uint8))

class ImageRequest(BaseModel):
    imageBase64: str

@app.post("/api/remove-bg")
async def remove_background(request: ImageRequest):
    try:
        # Download model if not exists
        await download_model()
        
        # Get session
        sess = get_session()
        if sess is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        # Decode base64 image
        image_data = base64.b64decode(request.imageBase64)
        input_image = Image.open(io.BytesIO(image_data))
        
        # Get original size
        original_size = input_image.size
        
        # Preprocess
        img = preprocess(input_image)
        
        # Run inference
        pred = sess.run(None, {'input': img})[0]
        
        # Postprocess
        mask = postprocess(pred, original_size)
        
        # Apply mask to original image
        input_image = input_image.convert('RGBA')
        input_array = np.array(input_image)
        mask_array = np.array(mask)
        input_array[:, :, 3] = mask_array
        
        # Convert back to PIL and then to base64
        output_image = Image.fromarray(input_array)
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG", optimize=True)
        output_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"imageBase64": output_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing background: {str(e)}")

@app.get("/api/healthcheck")
async def read_root():
    model_ready = os.path.exists(MODEL_PATH)
    return {
        "status": "ok",
        "message": "Background Removal API is running",
        "model_ready": model_ready
    }
