from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rembg import remove
import base64
from fastapi.responses import JSONResponse
from pathlib import Path
import os

# Initialize FastAPI
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up model caching in /tmp for Vercel
if os.environ.get("VERCEL"):
    CACHE_DIR = Path("/tmp/.u2net")
else:
    CACHE_DIR = Path.home() / ".u2net"
    
MODEL_PATH = CACHE_DIR / "u2net.onnx"

def download_model():
    if not MODEL_PATH.exists():
        print("Downloading u2net.onnx model...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from rembg.session_factory import new_session
        new_session("u2net")
    print("Model is ready!")

# Download model at startup
download_model()

class ImageRequest(BaseModel):
    imageBase64: str

@app.post("/api/remove-bg")
async def remove_background(request: ImageRequest):
    try:
        image_data = base64.b64decode(request.imageBase64)
        output_data = remove(image_data)
        output_base64 = base64.b64encode(output_data).decode("utf-8")
        return JSONResponse(content={"imageBase64": output_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing background: {str(e)}")

@app.get("/api/healthcheck")
async def read_root():
    return {
        "status": "ok",
        "message": "Background Removal API is running"
    }
