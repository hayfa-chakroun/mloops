import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--train_path", type=str, help="Path to training data")
    parser.add_argument("--test_path", type=str, help="Path to test data")

    args = parser.parse_args()

    if args.prepare or args.train or args.evaluate:
        if not args.train_path or not args.test_path:
            raise ValueError("Please provide train and test file paths using --train_path and --test_path")

    if args.prepare:
        x_train, x_test, y_train, y_test = prepare_data(args.train_path, args.test_path)
        print("Data prepared.")

    if args.train:
        x_train, x_test, y_train, y_test = prepare_data(args.train_path, args.test_path)
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data(args.train_path, args.test_path)
        model = load_model()
        evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()

