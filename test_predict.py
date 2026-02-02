from lead_scorer import load_all_saved_models, _get_prediction_and_probs
from models import evaluate_saved_models

message = "Please share pricing and demo for your solution."

print("=== Testing all saved models ===\n")

# Load all saved models directly
models_loaded = load_all_saved_models()

if not models_loaded:
    print("No models found. Please run 'python run_train_all.py' to train models.")
else:
    print(f"Found {len(models_loaded)} models:\n")
    
    # Filter out cross-validation models (exclude lead_ prefixed models, keep lead_classifier_ models)
    filtered_models = [m for m in models_loaded if not m['name'].startswith('lead_')]

    if not filtered_models:
        print("No lead_ models found. Please run training scripts to create models.")
    else:
        print(f"Showing predictions for {len(filtered_models)} lead models:\n")

        # Print individual predictions for each filtered model
        for m in filtered_models:
            model_name = m['name']
            print(f"Model: {model_name}")
            print("-" * 60)

            try:
                pred, top_prob, probs = _get_prediction_and_probs(m['pipeline'], message)
                print(f"Message: {message}")
                print(f"Prediction: {pred}")
                if top_prob is not None:
                    print(f"Top Confidence: {top_prob:.2%}")
                else:
                    print(f"Top Confidence: N/A")

                if probs:
                    print("Probabilities:")
                    for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
                        print(f"  {label}: {prob:.2%}")
            except Exception as e:
                print(f"Error predicting: {e}")

            print()
    
    # Also show evaluation report
    print("\n=== Full Model Evaluation ===\n")
    evaluate_saved_models(csv_path='Data/csvfile.csv')
    