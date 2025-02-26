from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Charger le modèle entraîné
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Définir l'API
app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}
