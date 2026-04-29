# bert_predict.py
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from preprocess import clean_text

def predict_bert(text, return_scores=False):
    """
    Predict sentiment using the BERT model.
    
    Args:
        text (str): The text to analyze
        return_scores (bool): If True, return confidence scores too
        
    Returns:
        str or tuple: The predicted sentiment label, or a tuple of (label, scores)
    """
    try:
        # Check if model exists
        model_path = "models/bert_model"
        
        if not os.path.exists(model_path):
            print(f"Error: BERT model not found. Please train the model using bert_train.py.")
            return "Unknown" if not return_scores else ("Unknown", {})
        
        # Load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(model_path)
        
        # Set to evaluation mode
        model.eval()
        
        # Preprocess text
        processed_text = clean_text(text)
        
        if not processed_text:
            print("Warning: Input text is empty after preprocessing.")
            return "Unknown" if not return_scores else ("Unknown", {})
        
        # Tokenize input
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
            
        # Get probabilities using softmax
        probs = torch.nn.functional.softmax(predictions, dim=1).squeeze().numpy()
        
        # Map prediction to label
        class_idx = np.argmax(probs)
        label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        result = label_mapping[class_idx]
        
        if return_scores:
            scores = {
                "Negative": float(probs[0]),
                "Neutral": float(probs[1]),
                "Positive": float(probs[2])
            }
            return result, scores
        
        return result
        
    except Exception as e:
        print(f"Error in BERT prediction: {e}")
        return "Error" if not return_scores else ("Error", {})

if __name__ == "__main__":
    sample = input("Enter a review: ")
    prediction, scores = predict_bert(sample, return_scores=True)
    print(f"BERT Prediction: {prediction}")
    print(f"Confidence scores: {scores}")