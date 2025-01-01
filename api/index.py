from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rembg import remove, new_session
import base64
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import numpy as np
from PIL import Image
import io

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

# Initialize rembg session with u2net_human_seg model (smaller than default)
session = new_session("u2net_human_seg")

class ImageRequest(BaseModel):
    imageBase64: str

@app.post("/api/remove-bg")
async def remove_background(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.imageBase64)
        
        # Convert to PIL Image
        input_image = Image.open(io.BytesIO(image_data))
        
        # Remove background
        output_image = remove(
            input_image,
            session=session,
            post_process_mask=True,
        )
        
        # Convert back to base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        output_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"imageBase64": output_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing background: {str(e)}")

@app.get("/api/healthcheck")
async def read_root():
    return {
        "status": "ok",
        "message": "Background Removal API is running"
    }
