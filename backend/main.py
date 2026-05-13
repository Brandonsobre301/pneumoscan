from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys

# Add parent directory to sys.path to import predict.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import load_model, predict_image

app = FastAPI(title="Pneumonia Detection API")

# Setup CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = load_model('../resnet18_pneumonia.pth')

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
         raise HTTPException(status_code=400, detail="File provided is not an image.")
         
    # Save the uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run prediction
        result = predict_image(temp_path, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return result
