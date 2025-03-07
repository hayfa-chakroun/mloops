import json
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from elasticsearch import Elasticsearch
import joblib
import numpy as np
from model_pipeline import prepare_data, train_model, save_model, load_model  # Assurez-vous que les imports sont utilisés.

# Connexion à Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

if es.ping():
    print("✅ Connexion Elasticsearch réussie !")
else:
    print("❌ Impossible de se connecter à Elasticsearch !")

def log_to_elasticsearch(experiment_id, run_id, metrics):
    """Envoie les logs de MLflow à Elasticsearch pour stockage."""
    log_entry = {"experiment_id": experiment_id, "run_id": run_id, "metrics": metrics}

    try:
        response = es.index(index="mlflow-metrics", body=json.dumps(log_entry))
        print(f"✅ Log envoyé à Elasticsearch : {response}")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi des logs à Elasticsearch : {str(e)}")

# Définir le tracking URI MLflow
mlflow.set_tracking_uri("http://localhost:5005")
print(f"🔍 MLflow Tracking URI utilisé : {mlflow.get_tracking_uri()}")

# Définition des chemins des fichiers et modèles
MODEL_PATH = "model.pkl"
TRAIN_FILE_PATH = "churn-bigml-80.csv"
TEST_FILE_PATH = "churn-bigml-20.csv"

# Initialisation de l'API FastAPI
app = FastAPI()

# Charger le modèle au démarrage de l'API
try:
    current_model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    current_model = None

# Format des données d'entrée pour la prédiction
class ChurnInput(BaseModel):
    features: list

class RetrainParams(BaseModel):
    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

@app.get("/")
def home():
    """Endpoint pour tester si l'API fonctionne."""
    return {"message": "L'API FastAPI fonctionne !"}

@app.post("/predict")
def predict(data: ChurnInput):
    """Endpoint pour faire des prédictions avec le modèle."""
    try:
        print(f"📨 Données reçues : {data.features}")

        if current_model is None:
            raise HTTPException(status_code=500, detail="Modèle non chargé. Essayez de le réentraîner.")

        input_data = np.array(data.features).reshape(1, -1)
        prediction = current_model.predict(input_data)

        print(f"✅ Prédiction faite : {prediction[0]}")
        return {"prediction": int(prediction[0])}

    except Exception as e:
        print(f"❌ Erreur de prédiction : {e}")
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")

@app.post("/retrain_params")
def retrain_params(params: RetrainParams):
    """Endpoint pour réentraîner le modèle avec des hyperparamètres personnalisés."""
    try:
        print("🔄 Début du réentraînement du modèle avec hyperparamètres...")

        X_train, X_test, y_train, y_test = prepare_data(TRAIN_FILE_PATH, TEST_FILE_PATH)

        new_model = train_model(
            X_train,
            y_train,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
        )

        save_model(new_model, MODEL_PATH)

        current_model = new_model

        with mlflow.start_run() as run:
            mlflow.log_params(params.dict())
            mlflow.log_metric("accuracy", 0.95)  # Remplace avec l'accuracy réelle

            log_to_elasticsearch(
                experiment_id=mlflow.active_run().info.experiment_id,
                run_id=mlflow.active_run().info.run_id,
                metrics={"accuracy": 0.95},  # Remplace avec l'accuracy réelle
            )

        print("✅ Modèle réentraîné avec succès avec les nouveaux hyperparamètres !")
        return {"message": "Modèle réentraîné avec succès.", "params": params.dict()}

    except Exception as e:
        print(f"❌ Erreur lors du réentraînement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

@app.post("/retrain")
def retrain():
    """Endpoint pour réentraîner le modèle avec les hyperparamètres par défaut."""
    try:
        print("🔄 Début du réentraînement du modèle avec les paramètres par défaut...")

        X_train, X_test, y_train, y_test = prepare_data(TRAIN_FILE_PATH, TEST_FILE_PATH)

        new_model = train_model(
            X_train,
            y_train,
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
        )

        save_model(new_model, MODEL_PATH)

        current_model = new_model

        with mlflow.start_run() as run:
            mlflow.log_metric("accuracy", 0.95)  # Remplace avec la vraie accuracy
            log_to_elasticsearch(
                experiment_id=mlflow.active_run().info.experiment_id,
                run_id=mlflow.active_run().info.run_id,
                metrics={"accuracy": 0.95},  # Remplace avec la vraie accuracy
            )

        print("✅ Modèle réentraîné avec succès avec les paramètres par défaut !")
        return {"message": "Modèle réentraîné avec succès avec les paramètres par défaut."}

    except Exception as e:
        print(f"❌ Erreur lors du réentraînement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# Démarrer l'API FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)

