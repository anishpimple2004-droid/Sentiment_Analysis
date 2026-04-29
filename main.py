#main.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
from sentiment_analyzer import analyze_sentiment
import subprocess
import io
import contextlib
import sys
import re
import os
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime

# Keep existing imports but we'll use our new function
try:
    from cnn_predict import predict_cnn
    from bayesian_predict import predict_bayesian
    from bert_predict import predict_bert
    from ensemble_predict import predict_ensemble
    from roberta_predict import predict_roberta
except ImportError:
    # Handle the case where some models are not available
    pass

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = r'C:\Users\91835\Desktop\sentiment_analysis_project\upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def landing():
    """Render the landing page"""
    return render_template("landing_page.html")

def parse_terminal_output(output):
    """Parse the terminal output from sentiment analyzer"""
    # Extract sentiment
    sentiment_match = re.search(r"Sentiment: (\w+)", output)
    sentiment = sentiment_match.group(1).capitalize() if sentiment_match else "Unknown"
    
    # Extract confidence
    confidence_match = re.search(r"Confidence: ([\d\.]+)", output)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
    
    # Extract scores
    scores = {}
    scores_section = re.search(r"All scores:(.*?)---", output, re.DOTALL)
    if scores_section:
        score_lines = scores_section.group(1).strip().split('\n')
        for line in score_lines:
            parts = line.strip().split(':')
            if len(parts) == 2:
                label = parts[0].strip()
                score = float(parts[1].strip())
                scores[label.capitalize()] = score
    
    return sentiment, confidence, scores

@app.route("/analyzer", methods=["GET", "POST"])
def analyzer():
    """Render the main analyzer interface"""
    result = None
    details = None
    
    if request.method == "POST":
        try:
            # Check if image was uploaded
            if 'image' in request.files:
                file = request.files['image']
                if file and file.filename != '' and allowed_file(file.filename):
                    # Save the image
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{filename}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(filepath)
            
            text = request.form["text"]
            model_choice = request.form["model"]
            
            if model_choice == "Terminal":
                # Use our new function directly
                sentiment, confidence, scores = analyze_sentiment(text)
                result = sentiment
                details = {
                    "model": "Terminal",
                    "scores": scores,
                    "confidence": confidence
                }
            elif model_choice == "CNN":
                prediction, scores = predict_cnn(text, return_scores=True)
                result = prediction
                details = {
                    "model": "CNN",
                    "scores": scores
                }
            elif model_choice == "Bayesian":
                prediction, confidence = predict_bayesian(text, return_score=True)
                result = prediction
                details = {
                    "model": "Bayesian",
                    "confidence": round(confidence, 4)
                }
            elif model_choice == "BERT":
                prediction, scores = predict_bert(text, return_scores=True)
                result = prediction
                details = {
                    "model": "BERT",
                    "scores": scores
                }
            elif model_choice == "Ensemble":
                prediction, ensemble_details = predict_ensemble(text)
                result = prediction
                details = {
                    "model": "Ensemble",
                    "ensemble_details": ensemble_details
                }
            elif model_choice == "RoBERTa":
                prediction, scores = predict_roberta(text, return_scores=True)
                result = prediction
                details = {
                    "model": "RoBERTa",
                    "scores": scores
                }

        except Exception as e:
            result = "Error"
            details = {"error": str(e)}
    
    return render_template("index.html", result=result, details=details)

@app.route("/extract-text", methods=["POST"])
def extract_text():
    """Extract text from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image part"}), 400
            
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if file and allowed_file(file.filename):
            # Save the image
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Run the image_text.py script
            try:
                script_path = r'C:\Users\91835\Desktop\sentiment_analysis_project\image_text.py'
                result = subprocess.run(
                    [sys.executable, script_path, filepath],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Extract text from output
                extracted_text = result.stdout.strip()
                # Find the text between "--- EXTRACTED TEXT ---" and "---------------------"
                text_match = re.search(r"--- EXTRACTED TEXT ---\n\n(.*?)\n\n---------------------", 
                                      extracted_text, re.DOTALL)
                
                if text_match:
                    clean_text = text_match.group(1).strip()
                else:
                    # If pattern not found, take everything after the first marker
                    marker_pos = extracted_text.find("--- EXTRACTED TEXT ---")
                    if marker_pos >= 0:
                        clean_text = extracted_text[marker_pos + len("--- EXTRACTED TEXT ---"):].strip()
                    else:
                        clean_text = extracted_text
                
                return jsonify({
                    "text": clean_text,
                    "image_path": filepath
                })
                
            except subprocess.CalledProcessError as e:
                return jsonify({
                    "error": f"Text extraction failed: {e.stderr}"
                }), 500
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for sentiment prediction"""
    try:
        data = request.json
        text = data.get("text", "")
        model = data.get("model", "CNN")
        
        if model == "Terminal":
            sentiment, confidence, scores = analyze_sentiment(text)
            return jsonify({
                "prediction": sentiment,
                "confidence": confidence,
                "scores": scores
            })
        elif model == "CNN":
            prediction, scores = predict_cnn(text, return_scores=True)
            return jsonify({
                "prediction": prediction,
                "scores": scores
            })
        elif model == "Bayesian":
            prediction, confidence = predict_bayesian(text, return_score=True)
            return jsonify({
                "prediction": prediction,
                "confidence": confidence
            })
        elif model == "BERT":
            prediction, scores = predict_bert(text, return_scores=True)
            return jsonify({
                "prediction": prediction,
                "scores": scores
            })
        elif model == "Ensemble":
            # For ensemble, allow configuration of models and weights
            models = data.get("models", None)  # Default: use all models
            weights = data.get("weights", None)  # Default: equal weights
            
            prediction, details = predict_ensemble(text, models=models, weights=weights)
            return jsonify({
                "prediction": prediction,
                "details": details
            })
        elif model == "RoBERTa":
            prediction, scores = predict_roberta(text, return_scores=True)
            return jsonify({
                "prediction": prediction,
                "scores": scores
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/extract-pdf-text", methods=["POST"])
def extract_pdf_text():
    """Extract text from uploaded PDF using pdf_text.py"""
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No PDF part"}), 400

        file = request.files['pdf']

        if file.filename == '':
            return jsonify({"error": "No selected PDF file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Path to your pdf_text.py
            script_path = r'C:\Users\91835\Desktop\sentiment_analysis_project\pdf_text.py'

            result = subprocess.run(
                [sys.executable, script_path, filepath],
                capture_output=True,
                text=True,
                check=True
            )

            # Extract text between markers
            output = result.stdout
            match = re.search(r"--- EXTRACTED TEXT ---\n(.*?)\n---------------------", output, re.DOTALL)
            if match:
                extracted_text = match.group(1).strip()
            else:
                extracted_text = "No text extracted."

            return jsonify({
                "text": extracted_text,
                "pdf_path": filepath
            })

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Text extraction failed: {e.stderr}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)