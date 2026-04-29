# cnn_predict.py
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

def predict_cnn(text, return_scores=False):
    """
    Predict sentiment using the CNN model.
    
    Args:
        text (str): The text to analyze
        return_scores (bool): If True, return raw prediction scores too
        
    Returns:
        str or tuple: The predicted sentiment label, or a tuple of (label, scores)
    """
    try:
        # Check if model and tokenizer files exist
        model_path = "models/sentiment_cnn_model.h5"
        tokenizer_path = "models/tokenizer.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            print(f"Error: Model files not found. Please train the model using cnn_train.py.")
            return "Unknown"
        
        # Load saved model and tokenizer
        model = load_model(model_path)
        tokenizer = joblib.load(tokenizer_path)
        
        # Preprocess input text
        from preprocess import clean_text
        processed_text = clean_text(text)
        
        if not processed_text:
            print("Warning: Input text is empty after preprocessing.")
            return "Unknown" if not return_scores else ("Unknown", {})
        
        # Convert text to sequence and pad
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_seq = pad_sequences(sequence, maxlen=100)
        
        # Predict sentiment
        raw_prediction = model.predict(padded_seq, verbose=0)[0]
        class_idx = np.argmax(raw_prediction)
        
        # Map prediction to label
        label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        result = label_mapping[class_idx]
        
        if return_scores:
            scores = {
                "Negative": float(raw_prediction[0]),
                "Neutral": float(raw_prediction[1]),
                "Positive": float(raw_prediction[2])
            }
            return result, scores
        
        return result
        
    except Exception as e:
        print(f"Error in CNN prediction: {e}")
        return "Error" if not return_scores else ("Error", {})

if __name__ == "__main__":
    sample = input("Enter a review: ")
    prediction, scores = predict_cnn(sample, return_scores=True)
    print(f"CNN Prediction: {prediction}")
    print(f"Confidence scores: {scores}")