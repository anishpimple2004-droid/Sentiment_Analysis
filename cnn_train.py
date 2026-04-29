# cnn_train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os
import joblib

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset
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

# Ensure we have review and sentiment columns
required_columns = ["review", "sentiment"]
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Required column '{col}' not found in dataset.")
        exit(1)

# Map sentiment labels
print("Preprocessing data...")
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df["encoded_labels"] = df["sentiment"].map(label_mapping)

# Handle any NaN values in the encoded labels
if df["encoded_labels"].isnull().any():
    print("Warning: Some sentiment values couldn't be mapped. Check your data.")
    df = df.dropna(subset=["encoded_labels"])
    df["encoded_labels"] = df["encoded_labels"].astype(int)

# Preprocess text data
max_words = 10000  # Increased vocabulary size
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df["review"])

X_train = tokenizer.texts_to_sequences(df["review"])
X_train = pad_sequences(X_train, maxlen=100)

# Convert labels to categorical format
y_train = to_categorical(df["encoded_labels"], num_classes=3)

# Calculate class weights to handle imbalance
class_counts = df["encoded_labels"].value_counts().to_dict()
total = sum(class_counts.values())
class_weights = {class_id: total / (count * 3) for class_id, count in class_counts.items()}

print("Building model...")
# Define the improved CNN model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=200, input_length=100),
    
    # Multiple convolutional layers with different filter sizes
    Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
    Conv1D(filters=64, kernel_size=4, activation="relu", padding="same"),
    Conv1D(filters=64, kernel_size=5, activation="relu", padding="same"),
    
    GlobalMaxPooling1D(),
    Dense(128, activation="relu"),
    Dropout(0.5),  # Add dropout to prevent overfitting
    Dense(3, activation="softmax")  # 3 classes: Negative, Neutral, Positive
])

# Compile the model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train the model
print("Training model...")
model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Save the model and tokenizer
print("Saving model...")
model.save("models/sentiment_cnn_model.h5")
joblib.dump(tokenizer, "models/tokenizer.pkl")

print("CNN model trained and saved successfully!")