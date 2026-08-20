import io
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
#  SETUP & CONFIGURATION
# ==========================================
app = FastAPI(title="AI Receipt Summarizer API", version="1.0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CHARS = "-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
idx_to_char = {idx: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)


# ==========================================
# MODEL DEFINITIONS (Exact architecture as trained)
# ==========================================
class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.25)
        )
        self.rnn = nn.LSTM(1024, hidden_size, bidirectional=True, batch_first=True, num_layers=2, dropout=0.25)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv_out = self.cnn(x)
        b, c, h, w = conv_out.size()
        conv_out = conv_out.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv_out)
        return self.fc(rnn_out).permute(1, 0, 2)


class GNNSummarizer(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNSummarizer, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        return torch.sigmoid(self.classifier(x))


# ==========================================
#  GLOBAL MODEL LOADING
# ==========================================
print("Loading Models into Memory...")
ocr_net = CRNN(num_classes=NUM_CLASSES).to(device)
gnn_net = GNNSummarizer(in_channels=388, hidden_channels=64).to(device)  # 388 features (Layout-Aware)
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

OCR_WEIGHTS_PATH = 'weights/ocr_crnn_weights.pth'
GNN_WEIGHTS_PATH = 'weights/gnn_summarizer_weights.pth'

try:
    ocr_net.load_state_dict(torch.load(OCR_WEIGHTS_PATH, map_location=device))
    gnn_net.load_state_dict(torch.load(GNN_WEIGHTS_PATH, map_location=device))
    ocr_net.eval()
    gnn_net.eval()
    print(" Weights loaded successfully!")
except Exception as e:
    print(f" Error loading weights: {e}")


def decode_prediction(pred_tensor):
    pred_tensor = pred_tensor.squeeze(1)
    _, max_indices = torch.max(pred_tensor, dim=1)
    max_indices = max_indices.cpu().numpy()
    text, prev_idx = [], 0
    for idx in max_indices:
        if idx != 0 and idx != prev_idx:
            text.append(idx_to_char[idx])
        prev_idx = idx
    return ''.join(text)


# ==========================================
#  API ENDPOINTS & LOGIC
# ==========================================
@app.post("/analyze_receipt/")
async def analyze_receipt(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not read the image.")

        height, width = img.shape[:2]
        scale = 800 / height
        img_resized = cv2.resize(img, (int(width * scale), 800))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[1])

        extracted_data = []
        for x, y, w, h in bounding_boxes:
            if w > 8 and h > 8:
                cropped_word = gray[y:y + h, x:x + w]
                target_w, target_h = 256, 32
                scale_crop = min(target_w / w, target_h / h)
                new_w, new_h = max(1, int(w * scale_crop)), max(1, int(h * scale_crop))
                resized = cv2.resize(cropped_word, (new_w, new_h))

                padded = np.zeros((target_h, target_w), dtype=np.float32)
                start_y = (target_h - new_h) // 2
                padded[start_y:start_y + new_h, 0:new_w] = resized / 255.0

                tensor_word = torch.tensor(padded).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = ocr_net(tensor_word)
                word_text = decode_prediction(output).strip()

                if len(word_text) > 1:
                    extracted_data.append({"text": word_text, "box": [x, y, x + w, y + h]})

        if len(extracted_data) < 2:
            cv2.imwrite("debug_thresh.jpg", thresh)
            return JSONResponse({"status": "failed", "message": "Not enough text detected. Debug image saved."})
        sentences = [item["text"] for item in extracted_data]
        boxes = [item["box"] for item in extracted_data]

        max_x = max([b[2] for b in boxes]) + 1
        max_y = max([b[3] for b in boxes]) + 1

        spatial_feats = [[b[0] / max_x, b[1] / max_y, b[2] / max_x, b[3] / max_y] for b in boxes]
        text_features = embedder.encode(sentences)

        node_features = torch.cat([
            torch.tensor(text_features, dtype=torch.float),
            torch.tensor(spatial_feats, dtype=torch.float)
        ], dim=1).to(device)

        sim_matrix = cosine_similarity(text_features)
        edge_indices = []
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j and sim_matrix[i][j] > 0.2:
                    edge_indices.append([i, j])
        for i in range(len(sentences) - 1):
            edge_indices.extend([[i, i + 1], [i + 1, i]])

        if not edge_indices: edge_indices = [[i, i] for i in range(len(sentences))]
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(device)

        with torch.no_grad():
            h = gnn_net.gat1(node_features, edge_index)
            h = F.elu(h)
            h = gnn_net.gat2(h, edge_index)
            h = F.elu(h)
            scores = torch.sigmoid(gnn_net.classifier(h)).squeeze(1).cpu().numpy().tolist()

        scored_sentences = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)

        num_summary = max(3, int(len(sentences) * 0.4))

        summary = [
            {"confidence": round(float(score), 4), "text": text}
            for score, text in scored_sentences[:num_summary]
        ]

        return JSONResponse({
            "status": "success",
            "total_lines_detected": len(sentences),
            "summary": summary
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)