from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy-load model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "finGPT-audit-model"  # Replace with your fine-tuned model path

def get_model_and_tokenizer():
    global model, tokenizer
    if model is None or tokenizer is None:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        logger.info("Model and Tokenizer loaded successfully")
    return model, tokenizer

# Initialize FastAPI app
app = FastAPI()

# Request schema
class RedFlagRequest(BaseModel):
    sentence: str

@app.get("/")
def read_root():
    return {"message": "Welcome to FinGPT-Audit API 🚀"}

# Asynchronous endpoint (original)
@app.post("/predict")
def predict_red_flag(data: RedFlagRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_prediction, data.sentence)
    return {"message": "Prediction in progress"}

def process_prediction(sentence: str):
    try:
        model, tokenizer = get_model_and_tokenizer()
        input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
        output_ids = model.generate(input_ids, max_length=50)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Async Prediction: {prediction}")
    except Exception as e:
        logger.error(f"Async Prediction error: {e}")

# Synchronous endpoint for testing and immediate feedback
@app.post("/predict_sync")
def predict_sync(data: RedFlagRequest):
    try:
        model, tokenizer = get_model_and_tokenizer()
        input_ids = tokenizer.encode(data.sentence, return_tensors="pt").to(device)
        output_ids = model.generate(input_ids, max_length=50)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Synchronous Prediction for '{data.sentence}': {prediction}")
        return {"label": prediction}
    except Exception as e:
        logger.error(f"Synchronous Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")