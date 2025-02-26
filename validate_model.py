import mlflow
import pandas as pd
from mlflow.models import validate_serving_input, convert_input_example_to_serving_input
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Définir le tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# 1. Charger le modèle
model_uri = 'runs:/05ca4559ba374103b9131712412bd2a5/model'
loaded_model = mlflow.pyfunc.load_model(model_uri)

# 2. Préparer un exemple de données
input_example = pd.DataFrame({
    'State': [0],  # Encodé
    'International plan': [1],  # Encodé
    'Voice mail plan': [0],  # Encodé
    'Number vmail messages': [5],  # Colonne manquante
    'Total day minutes': [265.1],
    'Total day calls': [110],
    'Total eve minutes': [197.4],
    'Total eve calls': [99],
    'Total night minutes': [244.7],
    'Total night calls': [91],
    'Total intl minutes': [10.0],
    'Total intl calls': [3],
    'Total intl charge': [2.7],  # Colonne manquante
    'Customer service calls': [1]
})

# 3. Valider le modèle
serving_payload = convert_input_example_to_serving_input(input_example)
validate_serving_input(model_uri, serving_payload)
print("✅ Validation réussie ! Le modèle fonctionne correctement.")

# 4. Faire des prédictions
predictions = loaded_model.predict(input_example.to_numpy())
print("Prédictions :", predictions)
