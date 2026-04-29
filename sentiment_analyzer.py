# sentiment_analyzer.py - Modified to work both as standalone and imported module
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Initialize global variables for model and tokenizer
model = None
tokenizer = None
label_map = None

def load_model():
    """Load the model and tokenizer once"""
    global model, tokenizer, label_map
    
    # Path to your model files
    model_path = "./model_files"
    
    # Load tokenizer and model
    print("Loading sentiment analysis model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Set model to evaluation mode
    model.eval()
    
    # Get number of labels from model config
    num_labels = model.config.num_labels
    print(f"Model has {num_labels} sentiment classes")
    
    # Define label mapping
    if num_labels == 3:
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
    else:
        label_map = {i: f"class_{i}" for i in range(num_labels)}
    
    return True

def analyze_sentiment(text):
    """Analyze sentiment of given text and return results"""
    global model, tokenizer, label_map
    
    # Load model if not already loaded
    if model is None or tokenizer is None:
        if not load_model():
            return None, None, None
    
    try:
        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.numpy()[0]
        
        # Get predicted class
        predicted_class = probs.argmax()
        sentiment = label_map[predicted_class]
        confidence = probs[predicted_class]
        
        # Create scores dictionary
        scores = {label_map[i].capitalize(): float(score) for i, score in enumerate(probs)}
        
        return sentiment.capitalize(), confidence, scores
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None, None, None

def main():
    """Run the analyzer in interactive mode from terminal"""
    if not load_model():
        return
    
    # Interactive loop
    print("\n=== Sentiment Analysis Tool ===")
    print("Enter text to analyze its sentiment")
    print("Type 'exit' to quit\n")
    
    while True:
        # Get user input
        user_input = input("Your text: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Analyze sentiment
        sentiment, confidence, scores = analyze_sentiment(user_input)
        
        if sentiment:
            # Print results
            print("\nSentiment Analysis Result:")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.4f}")
            
            # Print all scores
            print("\nAll scores:")
            for label, score in scores.items():
                print(f"  {label.lower()}: {score:.4f}")
            
            print("-" * 50)

if __name__ == "__main__":
    main()