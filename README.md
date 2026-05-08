# 🔬 DermaScan — AI-Powered Skin Analyzer

DermaScan is an end-to-end AI application that classifies skin conditions and provides detailed, medical-style interpretations. It combines Convolutional Neural Networks (CNN) for image recognition with Large Language Models (LLM) for explainable insights.

## ✨ Core Features

- **Deep Learning Vision:** EfficientNetB3 backbone (Transfer Learning) classifying 16 different skin conditions.
- **Explainable AI (XAI):** Grad-CAM heatmaps showing exactly which skin regions the CNN focused on.
- **Generative AI Analysis:** Llama-3.3-70b (via Groq API) generates structured explanations, treatment recommendations, and active skincare ingredients.
- **Live Fact-Checking:** Tavily Search API pulls the latest medical articles related to each prediction.
- **Per-class Evaluation:** Reports per-class F1, Precision, Recall, and ROC-AUC for full transparency.

## 📊 Model Performance

| Metric | Value |
|---|---|
| Backbone | EfficientNetB3 (300×300) |
| Test Accuracy | ~85-88% (target) |
| F1 Weighted | ~83-87% |
| ROC-AUC Macro | ~87-92% |
| Loss Function | Focal Loss (γ=2.0) |

> Per-class metrics are saved to `models/per_class_metrics.json` after training and viewable in the **Performa Model** dashboard page.

## ⚠️ Known Limitations & Design Decisions

### Dataset Composition
This model is trained on **two heterogeneous datasets**:

| Dataset | Image Type | Classes | ~Size |
|---|---|---|---|
| HAM10000 | Dermoscopy (skin-contact camera) | 7 clinical lesions | ~10K |
| SD-198 (subset) | Clinical / photo | 9 dermatological conditions | ~6.5K |

**Known issue:** Dermoscopy images and standard clinical photos have fundamentally different pixel distributions (lighting, contrast, background). Combining them in a single softmax head is a **deliberate portfolio trade-off** — not a production-grade design choice.

**Why it was done:** To cover a broader range of conditions useful to everyday users, while demonstrating the ability to handle heterogeneous data and class imbalance — skills valued in ML engineering roles.

**Mitigations applied:**
- **Focal Loss** (γ=2.0) reduces majority-class dominance at the loss level
- **Class weight balancing** (sklearn) provides additional imbalance correction
- **Per-domain evaluation** — HAM10000 and SD-198 accuracy are reported *separately* in `models/domain_metrics.json`, making the domain gap visible and auditable
- **Per-class F1 / ROC-AUC** reported explicitly — not hidden behind aggregate weighted averages

**In production:** Two separate models with an image-type routing layer (dermoscopy detector → clinical photo detector → route to specialist model).

### Accuracy Interpretation
Raw accuracy on imbalanced multi-class problems **can be misleading**. Always refer to:
- Per-class F1 (Streamlit dashboard → *Performa Model* → *Performa Per-Kelas*)
- Per-domain breakdown (Streamlit → *Analisis Per-Domain Dataset*)
- Normalized confusion matrix (`models/confusion_matrix.png`)
- ROC-AUC per class (`models/per_class_metrics.json`)

## 🏗️ Architecture & Tech Stack

This project follows an ultra-lean, service-oriented architecture separating the ML inference engine from the UI.

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow ≥ 2.12, Keras, OpenCV |
| Backbone | EfficientNetB3 (ImageNet pretrained) |
| Loss | Focal Loss (imbalance-aware) |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Frontend UI | Streamlit, Plotly |
| LLM & Search | Groq API (Llama-3.3-70b), Tavily API |
| Containerization | Docker, Docker Compose |

## 🐳 Running with Docker (Recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A trained model file (`models/skin_model_best.keras`) and its metadata (`models/class_names.json`) — see [Training the Model](#4-train-the-model) below

### 1. Configure Environment Variables
Copy the example file and fill in your API keys:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 2. Build & Start All Services
```bash
docker compose up --build
```
> On first build, Docker will pull the base image and install all dependencies — this can take a few minutes. Subsequent builds are fast thanks to layer caching.

### 3. Access the Application
| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| Swagger API Docs | http://localhost:8000/docs |

### Useful Docker Commands
```bash
# Run in background (detached mode)
docker compose up -d

# View live logs
docker compose logs -f

# View logs of a specific service
docker compose logs -f api
docker compose logs -f streamlit

# Stop all services
docker compose down

# Rebuild images after code changes
docker compose up --build

# Remove containers, networks, and volumes
docker compose down -v
```

---

## 🚀 Running Locally (Without Docker)

### 1. Prerequisites
Python 3.10 or 3.11 is required. We recommend Conda:
```bash
conda create -n dermascan_env python=3.10 -y
conda activate dermascan_env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Train the Model
Generate the model weights and metadata by running the training script. This executes a 2-phase Transfer Learning strategy using **EfficientNetB3** with **Focal Loss** and saves `skin_model_best.keras`, class names, and comprehensive evaluation metrics into the `models/` directory.
```bash
python -m src.train
```
> Ensure your dataset is prepared and located at the path specified inside `src/train.py`.

**Training outputs** (all git-ignored, generated locally):
```
models/
├── skin_model_best.keras         # Final trained model weights
├── class_names.json              # Ordered list of class names
├── evaluation_metrics.json       # Aggregate accuracy, F1, ROC-AUC
├── per_class_metrics.json        # Per-class F1, precision, recall, AUC
├── domain_metrics.json           # HAM10000 vs SD-198 performance breakdown
└── confusion_matrix.png          # Normalized confusion matrix heatmap
```

### 5. Start the Application
Open two separate terminal windows.

**Terminal 1 — Backend API:**
```bash
uvicorn deployment.api.main:app --reload --port 8000
```

**Terminal 2 — Frontend UI:**
```bash
streamlit run deployment/streamlit_app/app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📁 Project Structure

```
DermaScan/
├── deployment/
│   ├── api/
│   │   ├── Dockerfile          # FastAPI container (TF ≥ 2.12)
│   │   └── main.py             # FastAPI app & /predict endpoint
│   └── streamlit_app/
│       ├── Dockerfile          # Streamlit container
│       └── app.py              # Streamlit UI
├── src/
│   ├── train.py                # 2-phase EfficientNetB3 + Focal Loss pipeline
│   ├── predict.py              # Standalone prediction script
│   └── utils/
│       ├── gradcam.py          # Grad-CAM heatmap generation
│       ├── groq_analyzer.py    # Groq LLM integration
│       ├── tavily_search.py    # Tavily web search integration
│       └── logger.py           # Structured logging
├── models/                     # ⚠️ Git-ignored — generated by src/train.py
├── data/                       # ⚠️ Git-ignored — never containerized
├── notebooks/                  # Jupyter exploration notebooks
├── docker-compose.yml          # Orchestrates API + Streamlit services
├── .dockerignore               # Excludes data & model weights from build context
├── .env.example                # Template for required environment variables
└── requirements.txt            # Full Python dependency list
```

---

*Disclaimer: This application is built strictly for Machine Learning portfolio and research purposes. It is not a substitute for professional medical diagnosis. Always consult a certified Dermatologist for skin health concerns.*