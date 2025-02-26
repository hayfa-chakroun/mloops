import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
    # Création du parseur d'arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--model_path", type=str, default="model.pkl", help="Chemin pour sauvegarder/le modèle")
    parser.add_argument("--train_path", type=str, help="Chemin du fichier de données d'entraînement")
    parser.add_argument("--test_path", type=str, help="Chemin du fichier de données de test")
    
    # Récupérer les arguments
    args = parser.parse_args()

    # Vérifier que les fichiers d'entraînement et de test sont fournis
    if not args.train_path or not args.test_path:
        raise ValueError("Les chemins des fichiers d'entraînement et de test doivent être fournis")

    # Préparer les données une seule fois
    x_train, x_test, y_train, y_test = prepare_data(train_path=args.train_path, test_path=args.test_path)

    # Préparation des données
    if args.prepare:
        print("Préparation des données...")
        print("Données préparées avec succès.")

    # Entraînement du modèle
    if args.train:
        print("Entraînement du modèle...")
        model = train_model(x_train, y_train)
        save_model(model, filename=args.model_path)
        print(f"Modèle entraîné et sauvegardé sous {args.model_path}.")

    # Évaluation du modèle
    if args.evaluate:
        print("Évaluation du modèle...")
        model = load_model(filename=args.model_path)
        evaluate_model(model, x_test, y_test)
    
if __name__ == "__main__":
    main()

