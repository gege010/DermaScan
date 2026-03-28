# 🔬 DermaScan — AI-Powered Skin Analyzer

DermaScan is an end-to-end, ultra-lean AI application designed to classify skin conditions and provide detailed, medical-style interpretations. It combines Convolutional Neural Networks (CNN) for image recognition with Large Language Models (LLM) for explainable insights.

## ✨ Core Features

- **Deep Learning Vision:** Utilizes an EfficientNetB0 backbone (Transfer Learning) to classify 16 different skin conditions with high accuracy.
- **Explainable AI (XAI):** Implements Grad-CAM to generate heatmaps, showing exactly which regions of the skin the CNN focused on to make its decision.
- **Generative AI Analysis:** Integrates Llama-3.3-70b (via Groq API) to generate easily digestible explanations, treatment recommendations, and active skincare ingredients based on the CNN's prediction.
- **Live Fact-Checking:** Uses Tavily Search API to pull the latest medical articles and references related to the predicted condition.

## 🏗️ Architecture & Tech Stack

This project strictly follows an ultra-lean architecture, separating the core ML inference engine from the user interface.

- **Deep Learning:** TensorFlow 2.10, Keras, OpenCV
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Frontend UI:** Streamlit, Plotly
- **LLM & Search:** Groq API, Tavily API

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.10 installed (required for TensorFlow 2.10 native Windows GPU support). We recommend using Conda:
```bash
conda create -n dermascan_env python=3.10 -y
conda activate dermascan_env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Train the Model
Generate the model weights and metadata by running the training script. This pipeline will automatically execute a 2-phase training strategy and save skin_model_best.keras alongside its metadata into the model/saved_model/ directory.

```bash
python -m model.train
```
(Note: Ensure your dataset is properly prepared and located at the path specified in model/train.py before running this command. Alternatively, you can download my pre-trained weights from [Insert Link Here] and place them in the saved_model folder to skip training).

### 5. Start the Application
You will need two terminal windows to run the separated services.

**Terminal 1 (Backend API):**
```bash
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 (Frontend UI):**
```bash
streamlit run streamlit_app/app.py
```
Open your browser and navigate to `http://localhost:8501`.

---
*Disclaimer: This application is built strictly for Machine Learning portfolio and research purposes. It is not a substitute for professional medical diagnosis. Always consult a certified Dermatologist for skin health concerns.*