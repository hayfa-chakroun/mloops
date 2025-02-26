import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow
import mlflow.sklearn

# Définir le tracking URI
mlflow.set_tracking_uri("http://172.17.0.1:5005")
def prepare_data(train_path=None, test_path=None):
    if train_path and test_path:
        data_train = pd.read_csv(train_path)
        data_test = pd.read_csv(test_path)
        data = pd.concat([data_train, data_test], ignore_index=True)
    else:
        raise ValueError("Please provide train and test file paths")
    
    # Encodage des variables catégorielles
    encoder = LabelEncoder()
    data['State'] = encoder.fit_transform(data['State'])
    data['International plan'] = encoder.fit_transform(data['International plan'])
    data['Voice mail plan'] = encoder.fit_transform(data['Voice mail plan'])
    data['Churn'] = encoder.fit_transform(data['Churn'])

    # Suppression de certaines colonnes inutiles
    data = data.drop(['Number vmail messages', 'Total day charge', 'Total eve charge', 
                      'Total night charge', 'Total intl charge'], axis=1)

    # Séparation des variables explicatives et cible
    X = data.drop(['Churn'], axis=1)
    y = data['Churn']

    # Séparation des données en train et test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalisation des données
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Sauvegarde du scaler
    joblib.dump(scaler, 'scaler.joblib')

    return x_train_scaled, x_test_scaled, y_train, y_test

def train_model(X_train, y_train, model_name="Random Forest", n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    smote_enn = SMOTEENN(random_state=42)
    
    # Sélection du modèle
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42
        )
    elif model_name == "SVM (RBF Kernel)":
        model = SVC(
            C=1000,
            kernel='rbf',
            random_state=42
        )
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            C=0.01,
            random_state=42
        )
    else:
        raise ValueError(f"Modèle non reconnu : {model_name}")

    pipeline = ImbPipeline(steps=[
        ('smote_enn', smote_enn),
        ('classifier', model)
    ])

    # Commencer une nouvelle expérience MLflow
    with mlflow.start_run():
        print(f"🎯 Entraînement du modèle {model_name} avec MLflow")

        # Enregistrer les hyperparamètres
        mlflow.log_param("model_name", model_name)
        if model_name == "Random Forest":
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
        elif model_name == "Decision Tree":
            mlflow.log_param("max_depth", max_depth)
        elif model_name == "SVM (RBF Kernel)":
            mlflow.log_param("C", 1000)
        elif model_name == "Logistic Regression":
            mlflow.log_param("C", 0.01)

        # Entraîner le modèle
        pipeline.fit(X_train, y_train)

        # Sauvegarder le modèle avec MLflow
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"✅ Modèle {model_name} enregistré avec MLflow")

    return pipeline

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

def save_model(model, filename="model.pkl"):
    joblib.dump(model, filename)

def load_model(filename="model.pkl"):
    return joblib.load(filename)
