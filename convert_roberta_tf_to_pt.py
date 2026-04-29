# convert_roberta_tf_to_pt.py

from transformers import TFRobertaForSequenceClassification

# Path where your TensorFlow-based model is saved
model_path = "./model_files"  # update if needed

def convert_tf_to_pt(model_path):
    try:
        print("Loading TensorFlow RoBERTa model...")
        tf_model = TFRobertaForSequenceClassification.from_pretrained(model_path, from_pt=False)

        print("Saving as PyTorch model...")
        tf_model.save_pretrained(model_path, saved_model=True)

        print("✅ Conversion successful. PyTorch model saved at:", model_path)
    except Exception as e:
        print("❌ Conversion failed:", e)

if __name__ == "__main__":
    convert_tf_to_pt(model_path)
