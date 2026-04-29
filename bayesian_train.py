# bayesian_train.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
try:
    # Try to load balanced dataset first
    if os.path.exists("data/balanced_train.csv"):
        df = pd.read_csv("data/balanced_train.csv")
        print("Using balanced training dataset")
    else:
        df = pd.read_csv("data/train.csv")
        print("Using regular training dataset")
except FileNotFoundError:
    print("Error: Training dataset not found. Run preprocess.py first.")
    exit(1)

# Check for required columns
if "cleaned_text" not in df.columns or "sentiment" not in df.columns:
    print("Error: Dataset missing required columns. Run preprocess.py first.")
    exit(1)

# Convert sentiment labels to binary (just for Bayesian model)
# Map 'neutral' to 'positive' for simplicity with binary classifier
df["binary_sentiment"] = df["sentiment"].map({"negative": 0, "neutral": 1, "positive": 1})

print("Preparing data...")
texts = df["cleaned_text"].fillna("").tolist()
labels = df["binary_sentiment"].values

print("Training model with hyperparameter tuning...")
# Create a pipeline with TF-IDF and Naive Bayes
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=10000)),
    ('classifier', MultinomialNB())
])

# Define hyperparameters to tune
parameters = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__min_df': [1, 2],
    'classifier__alpha': [0.1, 0.5, 1.0]
}

# Perform grid search to find best parameters
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(texts, labels)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get the best model
best_model = grid_search.best_estimator_

# Save model and vectorizer separately for compatibility
print("Saving model...")
vectorizer = best_model.named_steps['vectorizer']
classifier = best_model.named_steps['classifier']

joblib.dump(classifier, "models/bayesian_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Bayesian model trained and saved in models/bayesian_model.pkl")