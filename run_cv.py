from models import load_all_saved_models
from sklearn.model_selection import cross_val_score
import pandas as pd

# Load data
df = pd.read_csv('Data/csvfile.csv')
df = df.dropna(subset=['Message'])
X = df['Message'].values
y = df['Label'].values

# Load models
models = load_all_saved_models()

# Filter to show only unique model types (prefer lead_ prefixed models if available)
seen_types = set()
filtered_models = []
for m in models:
    base_name = m['name'].replace('lead_', '').replace('classifier_', '')
    if base_name not in seen_types:
        seen_types.add(base_name)
        filtered_models.append(m)

for m in filtered_models:
    name = m['name']
    pipeline = m['pipeline']
    print(f"\n=== Cross-Validation for {name} ===")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")