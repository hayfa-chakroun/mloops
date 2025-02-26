# Utiliser une image Python légère
FROM python:3.8-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier uniquement le fichier requirements.txt (optimisation cache)
COPY requirements.txt .

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Copier tout le reste du projet
COPY . .

# Étape 7 : Exposer le port utilisé par FastAPI
EXPOSE 8001

# Étape 8 : Définir la commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]


