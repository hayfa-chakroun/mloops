# Variables
PYTHON = python3
PIP = pip3
TRAIN_PATH = "/home/hayfa/chakroun-hayfa-DS6-ml_project/churn-bigml-80.csv"
TEST_PATH = "/home/hayfa/chakroun-hayfa-DS6-ml_project/churn-bigml-20.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.joblib"
DEPLOY_DIR = "deploy"

# Cible par défaut
all: install prepare train evaluate deploy

install:
	$(PIP) install -r requirements.txt

prepare:
	$(PYTHON) main.py --prepare --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

train:
	$(PYTHON) main.py --train --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)
	@if [ ! -f $(MODEL_PATH) ]; then echo "❌ Erreur : model.pkl n'a pas été généré !"; exit 1; fi

evaluate:
	$(PYTHON) main.py --evaluate --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

deploy: train evaluate
	mkdir -p $(DEPLOY_DIR)
	ls -ld $(DEPLOY_DIR)
	cp -v $(MODEL_PATH) $(DEPLOY_DIR)/
	cp -rv src $(DEPLOY_DIR)/
	cp -v requirements.txt $(DEPLOY_DIR)/
	@echo "✅ Déploiement terminé dans $(DEPLOY_DIR)"

# Lancer l'API
api:
	$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8001

clean:
	rm -rf __pycache__
	rm -f $(MODEL_PATH)
	rm -f $(SCALER_PATH)
	rm -rf $(DEPLOY_DIR)

# Docker commands
build:
	docker build -t hayfa_chakroun_4ds6_mlops .

run:
	docker run -d -p 8001:8001 hayfa_chakroun_4ds6_mlops

run-interactive:
	docker run -it --rm -p 8001:8001 hayfa_chakroun_4ds6_mlops /bin/bash

push:
	docker tag hayfa_chakroun_4ds6_mlops hayfa12/mlflow-fastapi:latest
	docker push hayfa12/mlflow-fastapi:latest

cleandocker:
	@read -p "⚠️  Supprimer toutes les images et volumes inutilisés ? (y/N) " confirm && [ "$$confirm" = "y" ] && docker system prune -af || echo "❌ Annulé."

logs:
	docker logs -f $$(docker ps -q -f ancestor=hayfa_chakroun_4ds6_mlops)

stop:
	@if [ -n "$$(docker ps -q -f ancestor=hayfa_chakroun_4ds6_mlops)" ]; then \
		echo "🛑 Arrêt des conteneurs..."; \
		docker stop $$(docker ps -q -f ancestor=hayfa_chakroun_4ds6_mlops); \
	fi
	@if [ -n "$$(docker ps -aq -f ancestor=hayfa_chakroun_4ds6_mlops)" ]; then \
		echo "🗑 Suppression des conteneurs arrêtés..."; \
		docker rm $$(docker ps -aq -f ancestor=hayfa_chakroun_4ds6_mlops); \
	fi

rebuild:
	- make stop
	make build
	make run

status:
	@docker ps -f ancestor=hayfa_chakroun_4ds6_mlops --format "ID: {{.ID}} | Nom: {{.Names}} | Ports: {{.Ports}} | Statut: {{.Status}}"

.PHONY: all install prepare train evaluate deploy clean api

