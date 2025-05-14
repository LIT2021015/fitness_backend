import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
from pathlib import Path
from src.utils.exercise_generator import generate_exercise_plan
from src.utils.meal_planner import generate_meal_plan

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "src" / "data" / "models"

class UserInput(BaseModel):
    age: int
    weight: float
    height: float
    gender: str
    activity_level: str
    goal: str

models_cache = {}

def load_model(target):
    """Lazy load models on demand."""
    if target not in models_cache:
        model_path = MODEL_DIR / f"{target}_model.pkl"
        if not model_path.exists():
            logging.error(f"Model file for {target} not found at {model_path}")
            raise HTTPException(status_code=500, detail=f"Model file for {target} not found")
        with open(model_path, 'rb') as f:
            models_cache[target] = pickle.load(f)
    return models_cache[target]

def load_preprocessing():
    """Lazy load the preprocessing pipeline."""
    preprocessing_path = MODEL_DIR / "preprocessing.pkl"
    if not preprocessing_path.exists():
        logging.error(f"Preprocessing pipeline not found at {preprocessing_path}")
        raise HTTPException(status_code=500, detail="Preprocessing pipeline not found")
    
    with open(preprocessing_path, 'rb') as f:
        return pickle.load(f)

@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}

@app.get("/health")
def health_check():
    """Health check to ensure the app is running."""
    return {"status": "OK"}

@app.post("/api/fitness-plan")
async def get_fitness_plan(user_input: UserInput):
    logging.info("Received POST request at /api/fitness-plan")
    try:
        preprocessing = load_preprocessing()

        input_data = pd.DataFrame([user_input.dict()])
        logging.debug(f"Input Data:\n{input_data}")

        try:
            X_transformed = preprocessing.transform(input_data)
        except Exception as e:
            logging.error(f"Error during preprocessing transform: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed during preprocessing")

        try:
            numeric_features = ['age', 'weight', 'height']
            cat_features = ['gender', 'activity_level', 'goal']
            encoder = preprocessing.transformers_[1][1].named_steps['encoder']
            cat_feature_names = encoder.get_feature_names_out(cat_features)
            all_feature_names = numeric_features + list(cat_feature_names)
        except Exception as e:
            logging.error(f"Error extracting feature names: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to extract feature names")

        X_input_df = pd.DataFrame(X_transformed, columns=all_feature_names)

        predictions = {}
        try:
            for target in ['target_calories', 'protein_ratio', 'carb_ratio', 'fat_ratio', 'exercise_intensity']:
                model = load_model(target)
                predictions[target] = float(model.predict(X_input_df)[0])
        except Exception as e:
            logging.error(f"Prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Prediction failed")

        try:
            exercise_plan = generate_exercise_plan(predictions['exercise_intensity'])
        except Exception as e:
            logging.error(f"Error in exercise plan generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate exercise plan")

        try:
            meal_plan = generate_meal_plan(
                predictions['target_calories'],
                predictions['protein_ratio'],
                predictions['carb_ratio'],
                predictions['fat_ratio']
            )
        except Exception as e:
            logging.error(f"Error in meal plan generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate meal plan")

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
        raise http_exc
    except Exception as e:
        logging.error(f"Unhandled error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
