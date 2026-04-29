# bayesian_predict.py
import joblib
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources (with quiet flag to reduce output)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def predict_bayesian(text, return_score=False):
    """
    Predict sentiment using the Bayesian model.
    
    Args:
        text (str): The text to analyze
        return_score (bool): If True, return confidence score too
        
    Returns:
        str or tuple: The predicted sentiment label, or a tuple of (label, score)
    """
    try:
        # Check if model and vectorizer files exist
        model_path = "models/bayesian_model.pkl"
        vectorizer_path = "models/vectorizer.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"Error: Model files not found. Please train the model using bayesian_train.py.")
            return "Unknown" if not return_score else ("Unknown", 0.0)
        
        # Load the trained model and vectorizer
        bayesian_model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Preprocess text with our enhanced function
        from preprocess import clean_text
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            print("Warning: Input text is empty after preprocessing.")
            return "Unknown" if not return_score else ("Unknown", 0.0)

        # Transform the text using the vectorizer
        transformed_text = vectorizer.transform([cleaned_text])
        
        # Get prediction and probability
        prediction = bayesian_model.predict(transformed_text)[0]
        probabilities = bayesian_model.predict_proba(transformed_text)[0]
        
        # Get confidence score (probability of the predicted class)
        confidence = max(probabilities)
        
        # Map prediction to label
        label_mapping = {0: "Negative", 1: "Positive"}
        result = label_mapping[prediction]
        
        if return_score:
            return result, confidence
        
        return result
        
    except Exception as e:
        print(f"Error in Bayesian prediction: {e}")
        return "Error" if not return_score else ("Error", 0.0)

# Example Usage
if __name__ == "__main__":
    sample = input("Enter a review: ")
    prediction, confidence = predict_bayesian(sample, return_score=True)
    print(f"Bayesian Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")