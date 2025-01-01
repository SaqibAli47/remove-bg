from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from fastapi.responses import JSONResponse
from rembg import remove, new_session
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rembg session with u2netp model (smallest model)
session = new_session("u2netp")

class ImageRequest(BaseModel):
    imageBase64: str

@app.post("/api/remove-bg")
async def remove_background(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.imageBase64)
        input_image = Image.open(io.BytesIO(image_data))
        
        # Remove background
        output_image = remove(
            input_image,
            session=session,
            alpha_matting=False,
            post_process_mask=False,
        )
        
        # Convert back to base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG", optimize=True)
        output_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"imageBase64": output_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing background: {str(e)}")

@app.get("/api/healthcheck")
async def read_root():
    return {
        "status": "ok",
        "message": "Background Removal API is running",
        "session_initialized": session is not None
    }
