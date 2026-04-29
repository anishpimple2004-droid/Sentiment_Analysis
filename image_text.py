import os
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

# Initialize device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

# Global model variables
processor = None
model = None

def load_models():
    """Load the SmolDocling model and processor"""
    global processor, model
    try:
        print("[INFO] Loading SmolDocling model...", flush=True)
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview", torch_dtype=dtype
        ).to(DEVICE)
        print("[INFO] SmolDocling model loaded successfully", flush=True)
        return True
    except Exception as e:
        print(f"[ERROR] Loading models: {e}", flush=True)
        return False

def process_image_with_model(image):
    """Process an image with the SmolDocling model to extract doctags"""
    global processor, model
    if processor is None or model is None:
        success = load_models()
        if not success:
            return "Error: Failed to load models"

    try:
        print("[INFO] Processing image with model...", flush=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page to docling."}
                ]
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt", truncation=True, padding=True).to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=8192)
            prompt_len = inputs.input_ids.shape[1]
            trimmed_ids = generated_ids[:, prompt_len:]
            doctags = processor.batch_decode(trimmed_ids, skip_special_tokens=False)[0].lstrip()

        print("[INFO] DocTags extracted from image", flush=True)
        return doctags
    except Exception as e:
        print(f"[ERROR] Processing image: {e}", flush=True)
        return f"Error processing image: {str(e)}"

def extract_text_from_image(image_path):
    """Extract text from an image using SmolDocling and convert to markdown"""
    try:
        # Load the image
        print(f"[INFO] Loading image from: {image_path}", flush=True)
        image = Image.open(image_path).convert("RGB")
        
        # Process the image
        doctags = process_image_with_model(image)
        
        # Convert doctags to markdown
        print("[INFO] Converting DocTags to markdown...", flush=True)
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name="Image Document")
        doc.load_from_doctags(doctags_doc)
        markdown_text = doc.export_to_markdown()
        
        print("[INFO] Text extraction complete", flush=True)
        return markdown_text
    except FileNotFoundError:
        print(f"[ERROR] Image file not found: {image_path}", flush=True)
        return f"Error: Image file not found: {image_path}"
    except Exception as e:
        print(f"[ERROR] Extracting text: {e}", flush=True)
        return f"Error extracting text: {str(e)}"

def clean_markdown_text(markdown_text):
    """Clean the markdown text for better display in web interface"""
    # Remove markdown formatting that might interfere with sentiment analysis
    cleaned_text = markdown_text
    # Remove headers
    cleaned_text = ' '.join([line for line in cleaned_text.split('\n') if not line.strip().startswith('#')])
    # Remove bullet points
    cleaned_text = cleaned_text.replace('* ', '')
    cleaned_text = cleaned_text.replace('- ', '')
    # Remove formatting symbols
    cleaned_text = cleaned_text.replace('**', '')
    cleaned_text = cleaned_text.replace('__', '')
    cleaned_text = cleaned_text.replace('*', '')
    cleaned_text = cleaned_text.replace('_', '')
    # Remove extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

def main():
    # Check if image path is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python image_text.py <image_path>")
        print("Example: python image_text.py sample_image.jpg")
        return
    
    # Get the image path from command line arguments
    image_path = sys.argv[1]
    
    # Extract text from the image
    extracted_text = extract_text_from_image(image_path)
    cleaned_text = clean_markdown_text(extracted_text)
    
    # Print the extracted text to terminal in a format that can be easily extracted by main.py
    print("\n--- EXTRACTED TEXT ---\n")
    print(cleaned_text)
    print("\n---------------------\n")
    
    # Optionally save the extracted text to a file
    output_file = os.path.splitext(os.path.basename(image_path))[0] + "_extracted.txt"
    output_path = os.path.join(os.path.dirname(image_path), output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"[INFO] Extracted text saved to: {output_path}")

if __name__ == "__main__":
    main()