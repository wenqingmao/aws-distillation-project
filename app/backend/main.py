from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR_INSIDE_CONTAINER = "/app/mounted_model"
tokenizer = None
model = None
model_load_error = None

@app.on_event("startup")
async def load_model_on_startup():
    global tokenizer, model, model_load_error
    print(f"Attempting to load model. Device: {device}")
    if not os.path.exists(MODEL_DIR_INSIDE_CONTAINER) or not os.listdir(MODEL_DIR_INSIDE_CONTAINER):
        model_load_error = f"Model directory '{MODEL_DIR_INSIDE_CONTAINER}' is empty or does not exist. Check volume mount."
        print(f"ERROR: {model_load_error}")
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_INSIDE_CONTAINER)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_INSIDE_CONTAINER)
        model.to(device)
        model.eval()
        print(f"Model and tokenizer loaded successfully onto {device}.")
        print(f"Model config: {model.config}")
    except Exception as e:
        model_load_error = f"Error loading model/tokenizer: {str(e)}"
        print(f"CRITICAL ERROR: {model_load_error}")

class MessagePayload(BaseModel): # Keep this from Phase 1 for the /echo/ endpoint
    text: str

@app.get("/")
def read_root():
    if model and tokenizer:
        return {"message": f"Phase 2.A: FastAPI Backend with Model Loaded on {device}!"}
    elif model_load_error:
        return {"message": "FastAPI Backend - Model Loading FAILED.", "error": model_load_error}
    else:
        return {"message": "FastAPI Backend - Model is still loading or encountered an issue.", "model_loaded": False}

# Keep the simple echo endpoint from Phase 1 for now
@app.post("/echo/")
async def echo_message(payload: MessagePayload):
    return {"you_sent": payload.text, "backend_echoes": f"Backend (Phase 2.A) received: {payload.text}"}

# Simple endpoint to test model inference (very basic)
@app.post("/simple_predict/")
async def simple_predict(payload: MessagePayload):
    if not model or not tokenizer:
        return {"error": "Model not loaded. Cannot predict.", "details": model_load_error}
    try:
        inputs = tokenizer(payload.text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # For now, just return shape of logits or a simple message
        return {"input_text": payload.text, "model_output_type": str(type(outputs)), "logits_shape_if_present": str(outputs.logits.shape) if hasattr(outputs, 'logits') else "No logits attribute"}
    except Exception as e:
        return {"error": f"Simple prediction error: {str(e)}"}