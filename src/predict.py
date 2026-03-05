import joblib
import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'xgboost_model.pkl')

model = joblib.load(MODEL_PATH)
    

def predict(model, input_data):
    try:
        # Here you would typically preprocess the input_data to match the format expected by the model
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


def main():
    argument_parser = argparse.ArgumentParser(description="Load and use a trained model for prediction")
    argument_parser.add_argument("--input", type=str, required=True, help="Path to the input data file for prediction")
    args = argument_parser.parse_args()

    if model is None:
        return
    
    prediction = predict(model, args.input)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()