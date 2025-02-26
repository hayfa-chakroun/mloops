from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from model_pipeline import prepare_data, train_model, save_model, load_model
import mlflow
from elasticsearch import Elasticsearch
import json

# 🔍 Connexion à Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

if es.ping():
    print("✅ Connexion Elasticsearch réussie !")
else:
    print("❌ Impossible de se connecter à Elasticsearch !")

# 🔄 Fonction pour envoyer les logs MLflow vers Elasticsearch
def log_to_elasticsearch(experiment_id, run_id, metrics):
    log_entry = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "metrics": metrics
    }
    
    # Essayer d'envoyer à Elasticsearch et afficher la réponse
    try:
        response = es.index(index="mlflow-metrics", body=json.dumps(log_entry))
        print(f"✅ Log envoyé à Elasticsearch : {response}")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi des logs à Elasticsearch : {str(e)}")


# 🎯 Définition du tracking URI MLflow
mlflow.set_tracking_uri("http://localhost:5005")
print(f"🔍 MLflow Tracking URI utilisé : {mlflow.get_tracking_uri()}")

# 📂 Définition des chemins des fichiers et modèles
MODEL_PATH = "model.pkl"
TRAIN_FILE_PATH = "churn-bigml-80.csv"
TEST_FILE_PATH = "churn-bigml-20.csv"

# 🚀 Initialisation de l'API FastAPI
app = FastAPI()

# 🔄 Charger le modèle au démarrage de l'API
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None

# 📌 Définition du format des données en entrée
class ChurnInput(BaseModel):
    features: list

class RetrainParams(BaseModel):
    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

# 🏠 Endpoint pour tester si l'API fonctionne
@app.get("/")
def home():
    return {"message": "L'API FastAPI fonctionne !"}

# ✅ Endpoint pour faire des prédictions
@app.post("/predict")
def predict(data: ChurnInput):
    try:
        print(f"📨 Données reçues : {data.features}")

        # Vérifier que le modèle est bien chargé
        if model is None:
            raise HTTPException(status_code=500, detail="Modèle non chargé. Essayez de le réentraîner.")

        # Convertir les données en numpy array
        input_data = np.array(data.features).reshape(1, -1)

        # Faire la prédiction
        prediction = model.predict(input_data)

        print(f"✅ Prédiction faite : {prediction[0]}")
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        print(f"❌ Erreur de prédiction : {e}")
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")

# ✅ Endpoint pour réentraîner le modèle AVEC paramètres personnalisés
@app.post("/retrain_params")
def retrain_params(params: RetrainParams):
    try:
        print("🔄 Début du réentraînement du modèle avec hyperparamètres...")

        # Charger et préparer les données
        X_train, X_test, y_train, y_test = prepare_data(TRAIN_FILE_PATH, TEST_FILE_PATH)

        # Entraîner le modèle avec les hyperparamètres fournis
        new_model = train_model(X_train, y_train, 
                                n_estimators=params.n_estimators, 
                                max_depth=params.max_depth, 
                                min_samples_split=params.min_samples_split, 
                                min_samples_leaf=params.min_samples_leaf)

        # Sauvegarder le modèle mis à jour
        save_model(new_model, MODEL_PATH)

        # Mettre à jour le modèle en mémoire
        global model
        model = new_model

        # 🔥 Enregistrer les métriques dans MLflow
        with mlflow.start_run() as run:
            mlflow.log_params(params.dict())
            mlflow.log_metric("accuracy", 0.95)  # Modifier avec l'accuracy réelle

            log_to_elasticsearch(
                experiment_id=mlflow.active_run().info.experiment_id,
                run_id=mlflow.active_run().info.run_id,
                metrics={"accuracy": 0.95}  # Modifier avec l'accuracy réelle
            )

        print("✅ Modèle réentraîné avec succès avec les nouveaux hyperparamètres !")
        return {"message": "Modèle réentraîné avec succès.", "params": params.dict()}
    
    except Exception as e:
        print(f"❌ Erreur lors du réentraînement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# ✅ Endpoint pour réentraîner le modèle SANS paramètres (valeurs par défaut)
@app.post("/retrain")
def retrain():
    try:
        print("🔄 Début du réentraînement du modèle avec les paramètres par défaut...")

        # Charger et préparer les données
        X_train, X_test, y_train, y_test = prepare_data(TRAIN_FILE_PATH, TEST_FILE_PATH)

        # Entraîner le modèle avec des hyperparamètres par défaut
        new_model = train_model(X_train, y_train, n_estimators=100, max_depth=None, 
                                min_samples_split=2, min_samples_leaf=1)

        # Sauvegarder le modèle mis à jour
        save_model(new_model, MODEL_PATH)

        # Mettre à jour le modèle en mémoire
        global model
        model = new_model

        # 🔥 Enregistrer les métriques dans MLflow et Elasticsearch
        with mlflow.start_run() as run:
            mlflow.log_metric("accuracy", 0.95)  # Modifier avec la vraie accuracy
            log_to_elasticsearch(
                experiment_id=mlflow.active_run().info.experiment_id,
                run_id=mlflow.active_run().info.run_id,
                metrics={"accuracy": 0.95}  # Modifier avec la vraie accuracy
            )

        print("✅ Modèle réentraîné avec succès avec les paramètres par défaut !")
        return {"message": "Modèle réentraîné avec succès avec les paramètres par défaut."}
    
    except Exception as e:
        print(f"❌ Erreur lors du réentraînement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# 🚀 Démarrer l'API FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)

