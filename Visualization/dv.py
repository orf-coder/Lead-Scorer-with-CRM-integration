import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
from sklearn.metrics import confusion_matrix
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import load_all_saved_models, _predict_with_confidence
import pickle

# Data quality preprocessing function
def preprocess_data(df):
    """Clean and preprocess the dataset."""
    print(f"Original dataset size: {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates(subset=['Message', 'Label'])
    print(f"After duplicate removal: {len(df)}")

    # Clean text
    def clean_text(text):
        if pd.isna(text):
            return ""
        # Remove special characters, lowercase
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        return text.strip()

    df['Message'] = df['Message'].apply(clean_text)

    # Remove empty messages
    df = df[df['Message'].str.len() > 0]
    print(f"After text cleaning: {len(df)}")

    # Outlier detection - message length
    df['msg_length'] = df['Message'].str.len()
    q1 = df['msg_length'].quantile(0.25)
    q3 = df['msg_length'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df['msg_length'] >= lower_bound) & (df['msg_length'] <= upper_bound)]
    print(f"After outlier removal: {len(df)}")

    return df.drop('msg_length', axis=1)

# Load and preprocess datasets
csvfile_df = pd.read_csv('Data/csvfile.csv', on_bad_lines='skip', encoding='cp1252')
csvfile_df = preprocess_data(csvfile_df)

# Set up the plotting style
sns.set(style="whitegrid")

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Label distribution in csvfile.csv
csvfile_labels = csvfile_df['Label'].value_counts()
axes[0].bar(csvfile_labels.index, csvfile_labels.values, color=['blue', 'orange', 'green'])
axes[0].set_title('Label Distribution in csvfile.csv')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')

# Plot 2: Source distribution in csvfile.csv
csvfile_sources = csvfile_df['Source'].value_counts()
axes[1].bar(csvfile_sources.index, csvfile_sources.values, color='skyblue')
axes[1].set_title('Source Distribution in csvfile.csv')
axes[1].set_xlabel('Source')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()

# Save the basic plots
plt.savefig('data_visualization.png')

# Show the basic plots
plt.figure(1)
plt.show()

# Now create confusion matrix
# Load the ensemble model
try:
    with open('lead_classifier_ensemble.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Ensemble model not found. Please train the model first.")
    sys.exit(1)

# Predict on the csvfile.csv data
messages = csvfile_df['Message'].values
y_true = csvfile_df['Label'].values

# Predict
y_pred = model.predict(messages)

# Get unique labels from true data
unique_labels = sorted(list(set(y_true)))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('Confusion Matrix for Ensemble Model on csvfile.csv')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('Visualization/confusion_matrix.png')
plt.show()

print("Confusion matrix saved to Visualization/confusion_matrix.png")