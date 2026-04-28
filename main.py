import os
import cv2
import base64
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

# 1. Configuration & Setup
load_dotenv()

API_KEY = os.getenv("API_KEY", "secret-key")
MODEL_PATH = os.getenv("MODEL_PATH", "yolo26n-seg.pt")
DATA_PATH = os.getenv("DATA_PATH", "nutrition_dataset_large.csv")

# Global variables to store heavy objects in RAM
model = None
df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown. This is crucial for Render 
    to prevent reloading the model on every request (which causes crashes).
    """
    global model, df
    print(f"--- Booting up: Loading {MODEL_PATH} ---")
    
    # Load model and force to CPU
    model = YOLO(MODEL_PATH)
    model.to("cpu")
    
    # Load dataset safely
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            df["food_name"] = df["food_name"].astype(str).str.lower()
        else:
            print(f"Warning: Dataset not found at {DATA_PATH}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    yield
    # Cleanup on shutdown
    del model
    del df

app = FastAPI(title="Food Nutrition API", lifespan=lifespan)

# 2. Helper Logic
food_map = {
    "pizza_slice": "pizza", 
    "burger_big": "burger"
}

def get_nutrition(food_list):
    nutrition_list = []
    if df is None:
        return []
    for food in food_list:
        clean_name = food_map.get(food, food).lower()
        match = df[df["food_name"] == clean_name]
        if not match.empty:
            nutrition_list.append(match.iloc[0].to_dict())
    return nutrition_list

def calculate_total(nutrition_list):
    total = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}
    for item in nutrition_list:
        for key in total:
            # Ensure we handle potential NaN or missing values
            val = item.get(key, 0)
            total[key] += float(val) if pd.notnull(val) else 0.0
    return total

def verify_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# 3. API Endpoints
@app.get("/health")
def health_check():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "dataset_loaded": df is not None
    }

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key),
):
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # Resize for performance (Render CPU optimization)
        img_resized = cv2.resize(img, (320, 320))

        # Run YOLO Inference
        # imgsz=320 must match the resize above
        results = model.predict(img_resized, imgsz=320, verbose=False)
        
        # --- Create Annotated Image Overlay ---
        # plot() generates an image with bounding boxes, labels, and masks
        annotated_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Extract unique detected food labels
        detected_foods = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label not in detected_foods:
                    detected_foods.append(label)

        # Gather nutrition data
        nutrition_data = get_nutrition(detected_foods)
        total_stats = calculate_total(nutrition_data)

        # Final Response
        return {
            "detected_foods": detected_foods,
            "nutrition_data": nutrition_data,
            "total": total_stats,
            "summary": "High calorie" if total_stats["calories"] > 700 else "Balanced",
            "image_overlay": f"data:image/jpeg;base64,{img_base64}"
        }

    except Exception as e:
        # Log the error for Render dashboard debugging
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))