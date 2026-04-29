import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

# =============================================
# CONFIG SECTION - MODIFY THIS PART
# =============================================

# Set the path to the model file here
MODEL_PATH = r"C:\Users\91835\Desktop\sentiment_analysis_project\models\roberta_sentiment_model.pt"
  # CHANGE THIS PATH!

# =============================================
# SENTIMENT ANALYZER CLASS
# =============================================

class SentimentAnalyzer:
    def __init__(self, model_path):
        """Initialize the sentiment analyzer with the trained model"""
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Load model architecture
        print("Loading model architecture...")
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
        
        # Load trained weights
        print(f"Loading model weights from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
        
        # Prepare model for inference
        self.model.to(self.device)
        self.model.eval()
        
        # Define sentiment labels (0=negative, 1=neutral, 2=positive)
        self.id_to_label = {0: "negative", 1: "neutral", 2: "positive"}
    
    def predict(self, text):
        """
        Analyze the sentiment of the input text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Contains sentiment prediction and confidence scores
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        # Get prediction from model
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Get predicted class (highest probability)
        predicted_class = int(probs.argmax())
        
        # Create confidence scores dictionary
        confidence_scores = {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2])
        }
        
        # Get predicted sentiment label
        prediction = self.id_to_label[predicted_class]
        
        return {
            "sentiment": prediction,
            "confidence_scores": confidence_scores
        }

# =============================================
# TESTING FUNCTION
# =============================================

def test_sentiment_model():
    """Run interactive testing of the sentiment model"""
    print("\n" + "="*50)
    print(" ROBERTA SENTIMENT ANALYSIS MODEL TESTER ")
    print("="*50)
    
    try:
        # Initialize the analyzer
        analyzer = SentimentAnalyzer(MODEL_PATH)
        
        print("\n✓ Model initialization successful!")
        print("\nYou can now enter text to analyze its sentiment.")
        print("Type 'exit' or 'quit' to end the program.\n")
        
        # Main testing loop
        while True:
            # Get input from user
            user_input = input("\nEnter text to analyze: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                print("\nExiting program. Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input.strip():
                print("Please enter some text to analyze.")
                continue
            
            # Get prediction
            result = analyzer.predict(user_input)
            
            # Display results
            print("\n----- RESULTS -----")
            print(f"Sentiment: {result['sentiment'].upper()}")
            print("\nConfidence Scores:")
            for label, score in result['confidence_scores'].items():
                # Create a visual bar to represent the confidence score
                bar_length = int(score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                percentage = score * 100
                print(f"  {label.capitalize():8}: {bar} {percentage:.1f}%")
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check that:")
        print("1. The MODEL_PATH is set correctly")
        print("2. Required libraries are installed")
        print("3. The model file is a valid RoBERTa model trained for sentiment analysis")

# =============================================
# MAIN EXECUTION
# =============================================

if __name__ == "__main__":
    print("\nRoBERTa Sentiment Analysis Testing Tool")
    print("---------------------------------------")
    print(f"Model path: {MODEL_PATH}")
    
    # Check if model path has been changed from default
    if "path/to/" in MODEL_PATH:
        print("\n⚠  WARNING: You need to update the MODEL_PATH variable!")
        print("Please edit this file and set the correct path to your .pt file.")
        choice = input("\nContinue anyway? (y/n): ")
        if choice.lower() != 'y':
            print("Exiting program. Update the MODEL_PATH and try again.")
            exit()
    
    # Run the test
    test_sentiment_model()