import os
import cv2
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

load_dotenv()

API_KEY = os.getenv("API_KEY", "secret-key")
MODEL_PATH = os.getenv("MODEL_PATH", "yolo26n-seg.pt")
DATA_PATH = os.getenv("DATA_PATH", "nutrition_dataset_large.csv")

# Global variables for memory management
model = None
df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data on startup to avoid overhead during requests."""
    global model, df
    print("Loading model and dataset...")
    
    # Load model to CPU explicitly
    model = YOLO(MODEL_PATH)
    model.to("cpu")
    
    # Load dataset - only keep necessary columns to save RAM
    try:
        df = pd.read_csv(DATA_PATH)
        df["food_name"] = df["food_name"].astype(str).str.lower()
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    yield
    # Cleanup on shutdown
    del model
    del df

app = FastAPI(title="Food Nutrition API", lifespan=lifespan)

food_map = {"pizza_slice": "pizza", "burger_big": "burger"}

def get_nutrition(food_list):
    nutrition_list = []
    if df is None: return []
    for food in food_list:
        food = food_map.get(food, food)
        match = df[df["food_name"] == food.lower()]
        if not match.empty:
            nutrition_list.append(match.iloc[0].to_dict())
    return nutrition_list

def calculate_total(nutrition_list):
    total = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}
    for item in nutrition_list:
        for key in total:
            total[key] += float(item.get(key, 0))
    return total

def verify_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key),
):
    try:
        image_bytes = await image.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image_data = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image_data is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Resize to 320 for much faster inference on Render's weak CPU
        image_data = cv2.resize(image_data, (320, 320))

        # Perform inference
        results = model.predict(image_data, imgsz=320, verbose=False)
        
        detected_foods = list(set([model.names[int(box.cls[0])] for r in results for box in r.boxes]))
        
        nutrition_data = get_nutrition(detected_foods)
        total = calculate_total(nutrition_data)

        return {
            "detected_foods": detected_foods,
            "nutrition_data": nutrition_data,
            "total": total,
            "summary": "High calorie" if total["calories"] > 700 else "Balanced"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))