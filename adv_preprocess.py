# preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

def clean_text(text):
    # Handle None or empty text
    if not isinstance(text, str) or not text.strip():
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Handle contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'d", " would", text)
    
    # Remove special characters but preserve negation and some punctuation
    text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords but KEEP negation words
    stop_words = set(stopwords.words("english")) - {'no', 'not', 'nor', 'neither', 'never', 'none'}
    tokens = [word for word in tokens if word not in stop_words]
    
    # Return joined text
    return " ".join(tokens)

def preprocess_data():
    # Load your dataset (make sure it has "review" and "sentiment" columns)
    try:
        df = pd.read_csv("data/dataset.csv")
        df["cleaned_text"] = df["review"].apply(clean_text)
        
        # Split data into training and testing sets (80/20 split)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df.to_csv("data/train.csv", index=False)
        test_df.to_csv("data/test.csv", index=False)
        print("Data preprocessing complete. 'train.csv' and 'test.csv' saved in data/ folder.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

def create_balanced_dataset():
    try:
        # Load your dataset
        df = pd.read_csv("data/dataset.csv")
        df["cleaned_text"] = df["review"].apply(clean_text)
        
        # Get count of each class
        class_counts = df["sentiment"].value_counts()
        min_count = min(class_counts)
        
        # Create balanced dataset by sampling
        balanced_df = pd.DataFrame()
        for sentiment in df["sentiment"].unique():
            sampled = df[df["sentiment"] == sentiment].sample(min_count, random_state=42)
            balanced_df = pd.concat([balanced_df, sampled])
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into train/test
        train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)
        
        # Save balanced datasets
        train_df.to_csv("data/balanced_train.csv", index=False)
        test_df.to_csv("data/balanced_test.csv", index=False)
        print("Balanced data created. Files saved as 'balanced_train.csv' and 'balanced_test.csv'")
    except Exception as e:
        print(f"Error creating balanced dataset: {e}")

if __name__ == "__main__":
    preprocess_data()
    create_balanced_dataset()