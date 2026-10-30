# 🧠 Multi-Model Sentiment Analysis & Document Processing Platform

An end-to-end, multi-model Sentiment Analysis platform featuring deep learning architectures, Transformer models (BERT, RoBERTa), and classical machine learning, coupled with multimodal text extraction from PDFs and images via Vision-Language models.

---

## 📌 Project Overview

This platform provides sentiment classification across multiple state-of-the-art architectures and machine learning paradigms:
- **Transformer-based models**: Fine-tuned BERT and RoBERTa for deep contextual sentiment understanding.
- **Deep Learning**: 1D Convolutional Neural Network (CNN) for sequence modeling.
- **Probabilistic Modeling**: Bayesian (Naive Bayes) classifier for fast inference and probabilistic baselines.
- **Ensemble Engine**: Weighted voting mechanism combining predictions from multiple models for higher accuracy.
- **Multimodal Text Extraction**: Text extraction from images (powered by Hugging Face `SmolDocling`) and multi-page PDFs (via `PyMuPDF`).
- **Interactive Web Interface**: Modern, dark-themed responsive UI built on Flask and Bootstrap 5 with real-time prediction and score breakdown.
- **RESTful API**: Endpoints for programmatic inference and batch document processing.

---

## ✨ Features

- **Multi-Model Inference**:
  - **BERT**: Fine-tuned sequence classification with softmax confidence distribution.
  - **RoBERTa**: Optimized transformer pipeline for contextual understanding.
  - **CNN**: Word embeddings with 1D convolutions and pooling.
  - **Naive Bayes**: Fast statistical baseline with confidence estimation.
  - **Ensemble Classifier**: Combines CNN and BERT weighted probabilities to deliver consensus sentiment.
- **Multimodal Document Ingestion**:
  - **Image Text Extraction**: Extracts text from screenshots, documents, and photos using Vision2Seq models (`ds4sd/SmolDocling-256M-preview`).
  - **PDF Parser**: Extracts native text and embedded page images using `PyMuPDF`.
- **Dual Interface**:
  - **Web Dashboard**: Interactive Flask app with dedicated landing page, analyzer interface, and confidence visualizations.
  - **Interactive CLI**: Terminal-based interactive shell for rapid local testing.

---

## 🗂️ Project Structure

```text
sentiment_analysis_project/
├── main.py                     # Flask web server and routing
├── sentiment_analyzer.py       # Standalone interactive sentiment engine
├── bert_train.py               # BERT model training pipeline
├── bert_predict.py             # BERT inference module
├── cnn_train.py                # CNN model training script
├── cnn_predict.py              # CNN inference module
├── bayesian_train.py           # Naive Bayes training script
├── bayesian_predict.py         # Naive Bayes inference module
├── roberta_test.py             # RoBERTa evaluation & test script
├── convert_roberta_tf_to_pt.py # Checkpoint format converter
├── ensemble_predict.py         # Weighted ensemble classifier
├── image_text.py               # Vision-based OCR & text extraction (SmolDocling)
├── pdf_text.py                 # PDF document parser (PyMuPDF)
├── preprocess.py               # Text cleaning and preprocessing utilities
├── adv_preprocess.py           # Advanced NLP preprocessing pipeline
├── requirements.txt            # Python dependencies
├── templates/
│   ├── landing_page.html       # Landing page template
│   └── index.html              # Main sentiment analyzer dashboard
└── README.md                   # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/anishpimple2004-droid/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional for OCR / Vision features)*:
```bash
pip install PyMuPDF docling-core Pillow
```

---

## 💻 Usage

### Running the Web Application
```bash
python main.py
```
Open your browser and navigate to:
```
http://localhost:5000/
```

### Running the CLI Analyzer
```bash
python sentiment_analyzer.py
```

### Running Ensemble Inference
```bash
python ensemble_predict.py
```

---

## 🌐 API Reference

### 1. Analyze Sentiment
- **Endpoint**: `POST /api/predict`
- **Request Body**:
  ```json
  {
    "text": "The performance and user experience of this product are phenomenal!",
    "model": "BERT"
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "Positive",
    "scores": {
      "Negative": 0.012,
      "Neutral": 0.045,
      "Positive": 0.943
    }
  }
  ```

### 2. Extract Text from Document / Image
- **Endpoint**: `POST /extract-text`
- **Form Data**: `image` (file: `.png`, `.jpg`, `.pdf`)
- **Response**:
  ```json
  {
    "text": "Extracted text content...",
    "image_path": "upload/sample.png"
  }
  ```

---

## 🛠️ Tech Stack

- **Deep Learning & NLP**: PyTorch, Hugging Face Transformers, TensorFlow / Keras, Scikit-learn
- **Vision & Document AI**: SmolDocling, PyMuPDF (fitz), Pillow
- **Web Framework**: Flask, Jinja2, Bootstrap 5
- **Data & Utilities**: NumPy, Pandas, Joblib

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
