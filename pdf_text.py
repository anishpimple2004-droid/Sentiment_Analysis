import os
import sys
import fitz  # PyMuPDF
import re
from PIL import Image
import io
import tempfile

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF (fitz)"""
    try:
        print(f"[INFO] Loading PDF from: {pdf_path}", flush=True)
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        if pdf_document.page_count == 0:
            print("[ERROR] PDF has no pages", flush=True)
            return "Error: PDF has no pages"
        
        print(f"[INFO] Processing PDF with {pdf_document.page_count} pages...", flush=True)
        
        # Extract text from all pages
        extracted_text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            extracted_text += page.get_text()
            print(f"[INFO] Processed page {page_num + 1}/{pdf_document.page_count}", flush=True)
        
        # Clean the extracted text
        cleaned_text = clean_text(extracted_text)
        
        print("[INFO] Text extraction complete", flush=True)
        return cleaned_text
    
    except FileNotFoundError:
        print(f"[ERROR] PDF file not found: {pdf_path}", flush=True)
        return f"Error: PDF file not found: {pdf_path}"
    except Exception as e:
        print(f"[ERROR] Extracting text from PDF: {e}", flush=True)
        return f"Error extracting text from PDF: {str(e)}"

def extract_images_from_pdf(pdf_path):
    """Extract images from a PDF file and process them using the image text extractor"""
    try:
        # Only import if needed to avoid circular dependencies
        from image_text import extract_text_from_image
        
        print(f"[INFO] Extracting images from PDF: {pdf_path}", flush=True)
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        all_text = []
        image_count = 0
        
        # Process each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Get images from page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                # Get image info
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Save image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path)
                
                # Extract text from image
                print(f"[INFO] Processing image {img_index + 1} from page {page_num + 1}", flush=True)
                image_text = extract_text_from_image(temp_path)
                
                if not image_text.startswith("Error"):
                    all_text.append(image_text)
                    image_count += 1
                
                # Clean up temporary file
                os.unlink(temp_path)
        
        if image_count == 0:
            print("[INFO] No processable images found in PDF", flush=True)
            return ""
        
        combined_text = "\n\n".join(all_text)
        print(f"[INFO] Extracted text from {image_count} images in PDF", flush=True)
        return combined_text
    
    except Exception as e:
        print(f"[ERROR] Processing PDF images: {e}", flush=True)
        return f"Error processing PDF images: {str(e)}"

def clean_text(text):
    """Clean the extracted text for better display and analysis"""
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text)
    
    # Remove form feed characters
    cleaned = cleaned.replace('\f', ' ')
    
    # Remove redundant newlines while preserving paragraph breaks
    cleaned = re.sub(r'\n+', '\n', cleaned)
    
    # Remove other special characters that might interfere with analysis
    cleaned = re.sub(r'[^\w\s.,;:!?"\'-]', '', cleaned)
    
    # Remove extra whitespace
    cleaned = cleaned.strip()
    
    return cleaned

def main():
    # Check if PDF path is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python pdf_text.py <pdf_path> [--with-images]")
        print("Example: python pdf_text.py document.pdf")
        print("Add --with-images flag to also process images in the PDF")
        return
    
    # Get the PDF path from command line arguments
    pdf_path = sys.argv[1]
    process_images = "--with-images" in sys.argv
    
    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Optionally extract text from images in the PDF
    if process_images:
        image_text = extract_images_from_pdf(pdf_path)
        if image_text and not image_text.startswith("Error"):
            extracted_text += "\n\n--- EXTRACTED FROM IMAGES ---\n\n" + image_text
    
    # Print the extracted text to terminal in a format that can be easily extracted by main.py
    print("\n--- EXTRACTED TEXT ---\n")
    print(extracted_text.encode('cp1252', errors='replace').decode('cp1252'))
    print("\n---------------------\n")
    
    # Save the extracted text to a file
    output_file = os.path.splitext(os.path.basename(pdf_path))[0] + "_extracted.txt"
    output_path = os.path.join(os.path.dirname(pdf_path), output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    print(f"[INFO] Extracted text saved to: {output_path}")

if __name__ == "__main__":
    main()