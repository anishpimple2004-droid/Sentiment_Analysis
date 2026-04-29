# preprocess.py
import pandas as pd
import re
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nltk.download("punkt")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text, re.UNICODE)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

def preprocess_data():
    # Load your dataset (make sure it has "review" and "sentiment" columns)
    df = pd.read_csv("data/dataset.csv")
    df["cleaned_text"] = df["review"].apply(clean_text)
    
    # Split data into training and testing sets (80/20 split)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("Data preprocessing complete. 'train.csv' and 'test.csv' saved in data/ folder.")

if __name__ == "__main__":
    preprocess_data()
