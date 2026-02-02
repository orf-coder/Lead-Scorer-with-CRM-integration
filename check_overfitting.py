from models import LeadClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
df = pd.read_csv('Data/csvfile.csv')
df = df.dropna(subset=['Message'])
X = df['Message'].values
y = df['Label'].values

print("=== Overfitting Check: Training vs CV Accuracy (on csvfile.csv) ===\n")

model_types = ['logistic', 'svm', 'naive_bayes', 'ensemble']

for mtype in model_types:
    # Create and train model on csvfile.csv
    lc = LeadClassifier()
    lc.build_model(model_type=mtype)
    lc.model.fit(X, y)

    # Get training accuracy
    y_pred_train = lc.model.predict(X)
    train_accuracy = accuracy_score(y, y_pred_train)

    # Get CV accuracy
    cv_scores = cross_val_score(lc.model, X, y, cv=5, scoring='accuracy')
    cv_accuracy = cv_scores.mean()

    # Check for overfitting
    diff = train_accuracy - cv_accuracy
    overfitting = "YES" if diff > 0.1 else "NO"

    print(f"=== {mtype.upper()} ===")
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"CV Accuracy: {cv_accuracy:.3f}")
    print(f"Difference: {diff:.3f}")
    print(f"Overfitting: {overfitting}")
    print(f"CV Scores: {cv_scores}")
    print()