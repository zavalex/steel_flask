import joblib
from catboost import CatBoostRegressor
from pathlib import Path
import os

def load_model(model_path):
    file = open(model_path, 'rb')
    trained_model = joblib.load(file)
    return trained_model

def predict_with_trained_model(params):
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODEL_DIR = os.path.join(BASE_DIR, 'models\model.pkl')
    model = load_model(MODEL_DIR)
    pred = model.predict(params)
    return pred