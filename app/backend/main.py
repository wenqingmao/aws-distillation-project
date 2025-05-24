from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Ensure this is correct for your model
import torch
import os

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR_INSIDE_CONTAINER = "/app/mounted_model" # Path where model is mounted in container
tokenizer = None
model = None
model_load_error = None # To store any error during model loading

@app.on_event("startup")
async def load_model_on_startup():
    global tokenizer, model, model_load_error
    print(f"Attempting to load model for Phase 2.B. Device: {device}")
    if not os.path.exists(MODEL_DIR_INSIDE_CONTAINER) or not os.listdir(MODEL_DIR_INSIDE_CONTAINER):
        model_load_error = f"Model directory '{MODEL_DIR_INSIDE_CONTAINER}' is empty or does not exist. Check volume mount in docker-compose.yml."
        print(f"ERROR: {model_load_error}")
        return
    try:
        print(f"Loading tokenizer from: {MODEL_DIR_INSIDE_CONTAINER}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_INSIDE_CONTAINER)
        
        print(f"Loading model from: {MODEL_DIR_INSIDE_CONTAINER}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_INSIDE_CONTAINER)
        
        model.to(device) # Move model to GPU if available
        model.eval()     # Set model to evaluation mode
        print(f"Model and tokenizer loaded successfully onto {device}.")
        print(f"Model configuration: {model.config}") # Log model config for verification
    except Exception as e:
        model_load_error = f"CRITICAL Error loading model or tokenizer: {str(e)}"
        print(model_load_error)

class PredictionPayload(BaseModel):
    text: str

@app.get("/")
def read_root():
    if model and tokenizer:
        return {"message": f"FastAPI Model Backend (Phase 2.B) is running! Model loaded on {device}."}
    elif model_load_error:
        return {"message": "FastAPI Model Backend - Model Loading FAILED.", "error": model_load_error}
    else:
        return {"message": "FastAPI Model Backend - Model is initializing or encountered an issue.", "model_loaded": False}
    

@app.post("/predict/") # This is our main prediction endpoint now
async def predict(payload: PredictionPayload):
    if not model or not tokenizer:
        return {"error": "Model not loaded or not ready. Please check server logs or the /health endpoint.", "details": model_load_error}
    
    try:
        inputs = tokenizer(payload.text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        
        # Custom mapping for class IDs to desired output strings
        custom_id_to_label = {
            0: "No",
            1: "Maybe",
            2: "Yes"
        }
        
        # Use the custom mapping. Fallback if id is unexpected (shouldn't happen if model has 3 classes)
        predicted_string_label = custom_id_to_label.get(predicted_class_id, f"UNKNOWN_ID_{predicted_class_id}")

        return {
            "input_text": payload.text,
            "predicted_class_id": predicted_class_id,
            "predicted_label": predicted_string_label,
            "probabilities": probabilities.cpu().tolist()[0] # Send probabilities for all classes as a list
        }

    except Exception as e:
        print(f"Prediction endpoint error: {e}")
        # Consider logging the full traceback here for better debugging if needed
        # import traceback
        # print(traceback.format_exc())
        return {"error": f"Prediction endpoint error: {str(e)}"}

@app.get("/health")
def health_check():
    model_status = "loaded" if model and tokenizer else "not_loaded"
    model_error_details = model_load_error if model_load_error else "None"
    
    gpu_available_torch = torch.cuda.is_available()
    model_on_gpu_check = False
    if model and gpu_available_torch:
        try:
            model_on_gpu_check = next(model.parameters()).is_cuda
        except Exception: # Handle case where model might not have parameters or other issues
            pass

    return {
        "status": "healthy" if model_status == "loaded" else "unhealthy",
        "timestamp": torch.cuda.Event(enable_timing=True).record() if gpu_available_torch else None, # Just an example, might not be best
        "model_status": model_status,
        "model_load_error": model_error_details,
        "device_used_by_model": str(device) if model else "N/A (model not loaded)",
        "torch_cuda_available": gpu_available_torch,
        "model_on_gpu": model_on_gpu_check,
        "model_config_type": model.config.model_type if model else "N/A",
        "model_dir_in_container_exists": os.path.exists(MODEL_DIR_INSIDE_CONTAINER),
        "model_dir_in_container_empty": not os.listdir(MODEL_DIR_INSIDE_CONTAINER) if os.path.exists(MODEL_DIR_INSIDE_CONTAINER) else "N/A"
    }