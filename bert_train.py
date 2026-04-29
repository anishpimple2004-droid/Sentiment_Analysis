# bert_train.py
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Define dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_bert_model():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Map sentiment labels
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df["encoded_labels"] = df["sentiment"].map(label_mapping)
    
    # Handle any NaN values in the encoded labels
    if df["encoded_labels"].isnull().any():
        print("Warning: Some sentiment values couldn't be mapped. Check your data.")
        df = df.dropna(subset=["encoded_labels"])
        df["encoded_labels"] = df["encoded_labels"].astype(int)
    
    # Split data into train and validation sets
    texts = df["review"].fillna("").tolist()
    labels = df["encoded_labels"].values
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare datasets and dataloaders
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Load pre-trained model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,  # Negative, Neutral, Positive
        output_attentions=False,
        output_hidden_states=False
    )
    
    model = model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    
    # Set up training parameters
    epochs = 4
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Training BERT model...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        # Progress bar for training
        train_progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in train_progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Clear previous gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Update progress bar
            train_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        
        # Progress bar for validation
        val_progress_bar = tqdm(val_dataloader, desc="Validation")
        
        for batch in val_progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                val_accuracy += (predictions == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = val_accuracy / len(val_dataset)
        
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Save the trained model
    model_save_path = "models/bert_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"BERT model trained and saved to {model_save_path}")

if __name__ == "__main__":
    train_bert_model()