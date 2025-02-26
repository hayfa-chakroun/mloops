import streamlit as st
import requests

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8001/predict"

st.title("🧠 Prédiction du Churn")

# Interface utilisateur
st.write("Entrez les caractéristiques du client pour obtenir une prédiction.")

# Créer des entrées pour les features du modèle
features = [
    st.number_input("State (0-2)", min_value=0, max_value=2, value=0),
    st.number_input("Account length", value=100),
    st.number_input("Area code", value=415),
    st.number_input("International plan (0-1)", min_value=0, max_value=1, value=0),
    st.number_input("Voice mail plan (0-1)", min_value=0, max_value=1, value=1),
    st.number_input("Total day minutes", value=120.0),
    st.number_input("Total day calls", value=30),
    st.number_input("Total eve minutes", value=200.0),
    st.number_input("Total eve calls", value=50),
    st.number_input("Total night minutes", value=300.0),
    st.number_input("Total night calls", value=60),
    st.number_input("Total intl minutes", value=10.0),
    st.number_input("Total intl calls", value=3),
    st.number_input("Customer service calls", value=2)
]


if st.button("📊 Prédire"):
    # Construire la requête
    data = {"features": features}

    # Envoyer la requête à l'API FastAPI
    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        st.success(f"✅ Résultat : {'Churn' if prediction == 1 else 'Non Churn'}")
    else:
        st.error(f"⚠️ Erreur de l'API : {response.json()}")


