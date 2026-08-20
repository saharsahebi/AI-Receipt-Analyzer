#  Smart AI Receipt Analyzer

An end-to-end Full-Stack AI application designed to automatically read, analyze, and extract key information from receipt images. 

Unlike traditional OCR systems that rely solely on text matching, this project utilizes a **Layout-Aware Graph Neural Network (GNN)** to understand the spatial geometry and structure of the document, significantly boosting extraction accuracy.

---

##  Key Features
* **Custom OCR Engine:** A Convolutional Recurrent Neural Network (CRNN) built from scratch using PyTorch to recognize text from image crops.
* **Layout-Aware GNN:** Uses Graph Attention Networks (GAT) that combine text embeddings (SentenceTransformers) with spatial bounding box coordinates and custom heuristics to classify and extract crucial entities (e.g., Total Amount, Date, Company Name).
* **Robust Image Processing:** Utilizes OpenCV for adaptive thresholding, contour detection, and noise reduction.
* **Modern Web Interface:** A sleek, responsive React/Vite frontend for seamless user interaction.
* **Fast Backend API:** Powered by FastAPI for high-performance asynchronous request handling.

---

## ️ Architecture & Pipeline
1. **Input:** User uploads a receipt image via the React UI.
2. **Preprocessing (OpenCV):** The backend resizes, converts to grayscale, applies Gaussian blur, and uses adaptive thresholding to find word contours.
3. **OCR (CRNN):** Extracted word crops are fed into the CRNN to decode the pixels into text.
4. **Graph Construction:** Nodes are created for each detected sentence. Features consist of a **392-dimensional vector** (384D Text Embedding + 4D Normalized Bounding Box Coordinates + 4D Hand-crafted Heuristics like currency and date formats).
5. **Information Extraction (GNN):** The Graph Neural Network analyzes the relationships and layout, scoring each node to determine if it represents a key entity.
6. **Output:** The most confident results are returned as a JSON response and rendered beautifully on the frontend.

---

##  Tech Stack
**AI & Machine Learning:**
* PyTorch & PyTorch Geometric
* Sentence-Transformers (`all-MiniLM-L6-v2`)
* OpenCV & Scikit-Learn

**Backend:**
* Python 3 & FastAPI
* Uvicorn

**Frontend:**
* React (Vite)
* Axios
* Pure CSS (Modern UI/UX)

---

##  Dataset & Model Training
To ensure reproducibility and transparency, the complete training pipeline is publicly available.

* **Dataset:** The models were trained on the [ICDAR 2019 SROIE Dataset](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2), which contains thousands of scanned receipt images with highly detailed bounding box coordinates and text annotations.
* **Training Code:** The deep learning models (CRNN and Layout-Aware GNN) were built and trained from scratch. You can explore the exact training process, data augmentation, and evaluation metrics in my Kaggle notebook:
  * 🔗 [View the Training Source Code on Kaggle](https://www.kaggle.com/code/saharsahebi/model-training-pipeline)

---

##  Pre-trained Models (Hugging Face)
The fully trained PyTorch weights for this project are publicly available on Hugging Face. You can download them directly to run the end-to-end inference pipeline without needing to retrain the models from scratch.

* ** Hugging Face Repository:** [saharsahebi/layout-aware-receipt-ocr](https://huggingface.co/saharsahebi/layout-aware-receipt-ocr)

**Available Weights:**
* `ocr_crnn_weights.pth`: The trained Convolutional Recurrent Neural Network for text extraction (CER: 0.92%).
* `gnn_summarizer_weights.pth`: The Ultimate 392-D Layout-Aware Graph Neural Network for entity classification (F1-Score: 72.14%).

---
##  Application Demo

![Receipt Upload & Analysis](assets/1.png)
![Receipt Upload & Analysis](assets/2.png)
![Receipt Upload & Analysis](assets/3.png)

---

## Project Structure
```text
AI-Receipt-Analyzer/
├── backend/                           # Python FastAPI & AI Models
│   ├── main.py                        # API endpoints and inference logic
│   ├── requirements.txt               # Python dependencies
│   └── weights/                       # Saved model weights (.pth files)
│
├── frontend/                          # React + Vite UI
│   ├── src/                           # React components and CSS
│   ├── package.json                   # Node.js dependencies
│   └── vite.config.js         
│
├── model-training/           # Training codebase and evaluation metrics
│   └── model_training_pipeline.ipynb  # Jupyter notebook for CRNN and GNN training
│
└── README.md                          # Project documentation