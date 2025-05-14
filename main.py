import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
from pathlib import Path

from src.utils.exercise_generator import generate_exercise_plan
from src.utils.meal_planner import generate_meal_plan

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "src" / "data" / "models"

# Input model
class UserInput(BaseModel):
    age: int
    weight: float
    height: float
    gender: str
    activity_level: str
    goal: str

# Model loader
def load_model(target):
    model_path = MODEL_DIR / f"{target}_model.pkl"
    logging.info(f"Trying to load model from {model_path}")
    if not model_path.exists():
        logging.error(f"Model not found for {target} at {model_path}")
        raise HTTPException(status_code=500, detail=f"Model file for {target} not found")
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Preprocessing pipeline loader
def load_preprocessing():
    preprocessing_path = MODEL_DIR / "preprocessing.pkl"
    logging.info(f"Trying to load preprocessing from {preprocessing_path}")
    if not preprocessing_path.exists():
        logging.error(f"Preprocessing pipeline not found at {preprocessing_path}")
        raise HTTPException(status_code=500, detail="Preprocessing pipeline not found")
    with open(preprocessing_path, "rb") as f:
        return pickle.load(f)

# Root route
@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}

# Debug route to check model files
@app.get("/debug-models")
def debug_models():
    try:
        files = [f.name for f in MODEL_DIR.glob("*.pkl")]
        return {"models_found": files}
    except Exception as e:
        logging.error(f"Error listing model files: {e}")
        return {"error": str(e)}

# Main POST route
@app.post("/api/fitness-plan")
async def get_fitness_plan(user_input: UserInput):
    logging.info("Received POST request at /api/fitness-plan")
    try:
        # Load preprocessing
        logging.info("Loading preprocessing pipeline...")
        preprocessing = load_preprocessing()

        # Prepare input
        input_data = pd.DataFrame([user_input.dict()])
        logging.debug(f"Input DataFrame:\n{input_data}")

        # Transform input
        logging.info("Transforming input...")
        X_transformed = preprocessing.transform(input_data)

        # Extract feature names
        logging.info("Extracting feature names...")
        numeric_features = ['age', 'weight', 'height']
        cat_features = ['gender', 'activity_level', 'goal']
        encoder = preprocessing.transformers_[1][1].named_steps['encoder']
        cat_feature_names = encoder.get_feature_names_out(cat_features)
        all_feature_names = numeric_features + list(cat_feature_names)
        X_input_df = pd.DataFrame(X_transformed, columns=all_feature_names)

        # Predict
        logging.info("Generating predictions...")
        targets = ['target_calories', 'protein_ratio', 'carb_ratio', 'fat_ratio', 'exercise_intensity']
        predictions = {}
        for target in targets:
            model = load_model(target)
            predictions[target] = float(model.predict(X_input_df)[0])
            logging.debug(f"{target} prediction: {predictions[target]}")

        # Generate plans
        logging.info("Generating exercise plan...")
        exercise_plan = generate_exercise_plan(predictions['exercise_intensity'])

        logging.info("Generating meal plan...")
        meal_plan = generate_meal_plan(
            predictions['target_calories'],
            predictions['protein_ratio'],
            predictions['carb_ratio'],
            predictions['fat_ratio']
        )

        logging.info("Returning successful response.")
        return {
            "predictions": {
                "target_calories": round(predictions['target_calories']),
                "protein_ratio": round(predictions['protein_ratio'] * 100, 1),
                "carb_ratio": round(predictions['carb_ratio'] * 100, 1),
                "fat_ratio": round(predictions['fat_ratio'] * 100, 1),
                "exercise_intensity": round(predictions['exercise_intensity'], 1)
            },
            "exercise_plan": exercise_plan,
            "meal_plan": meal_plan
        }

    except HTTPException as http_exc:
        logging.error(f"HTTPException: {http_exc.detail}", exc_info=True)
        raise http_exc
    except Exception as e:
        logging.error(f"Unhandled error in /api/fitness-plan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
