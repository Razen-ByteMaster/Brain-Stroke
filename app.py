from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI()

# Load model data
try:
    model_data = joblib.load("stroke_model_best.pkl")
    model = model_data["model"]
    categorical_features = model_data["categorical_features"]
    numerical_features = model_data["numerical_features"]
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


# Define input schema (aliases let you use lowercase in JSON)
class StrokeInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str = Field(..., alias="residence_type")
    avg_glucose_level: float
    bmi: float
    smoking_status: str

    class Config:
        allow_population_by_field_name = True


def prepare_input(input_data: StrokeInput):
    """Prepare input DataFrame with correct column names"""
    # Use dict(by_alias=False) → gives field names as defined in the model
    df = pd.DataFrame([input_data.dict(by_alias=False)])
    return df


@app.post("/predict")
def predict_stroke(input_data: StrokeInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Prepare DataFrame
        input_df = prepare_input(input_data)

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Response
        result = "High stroke risk" if prediction == 1 else "Low stroke risk"
        return {
            "prediction": result,
            "probability": round(float(probability), 4),
            "stroke_risk": int(prediction),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/")
def health_check():
    return {"status": "OK", "model_loaded": model is not None}
