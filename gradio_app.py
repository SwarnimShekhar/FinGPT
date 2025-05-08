import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
import logging
from functools import lru_cache

# Setup logging
logging.basicConfig(
    filename="fingpt_audit_logs.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

# Caching model loading to avoid reloading it on each request
@lru_cache(maxsize=1)
def get_model():
    model_name_or_path = "finGPT-audit-model"  # local dir or Hugging Face model ID
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Inference function
def classify_text(text):
    if not text.strip():
        return "⚠️ Please enter financial text."

    input_text = "classify: " + text
    model, tokenizer, device = get_model()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(**inputs, max_length=10)
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Logging
    logging.info(f"Input: {text} | Prediction: {prediction}")
    return prediction.capitalize()

# Gradio Interface
interface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter financial statement (e.g. earnings, revenue, liabilities)..."
    ),
    outputs="text",
    title="📊 FinGPT-Audit",
    description="Detects sentiment (positive, negative, neutral) in financial disclosures using a fine-tuned FLAN-T5 model.",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(share=True)