import re
import string
import spacy
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorWithPadding
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load SpaCy NER model for preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Tokenize and lemmatize using SpaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

# Example financial text
example_text = "The revenue for Q1 2025 is $10 million, up by 5% from last year. Debt to equity ratio is at 1.2."
clean_text = preprocess_text(example_text)
print(clean_text)

# Load financial phrase dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")

# Load tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocessing function for the dataset
def preprocess(example):
    input_text = "classify: " + example["sentence"]

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    target_text = label_map[int(example["label"])]  # Convert label to text

    model_inputs = tokenizer(
        input_text,
        max_length=128,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        target_text,
        max_length=16,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["labels"],
    }

# Apply preprocessing and remove raw fields
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
tokenized_dataset.set_format("torch")

# Create DataCollator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Manually split the dataset into train, validation, and test
train_test = tokenized_dataset["train"].train_test_split(test_size=0.1)  # 10% for testing
train_valid = train_test["train"].train_test_split(test_size=0.1)  # 10% for validation

# Now you have train, validation, and test splits
train_dataset = train_valid["train"]
val_dataset = train_valid["test"]
test_dataset = train_test["test"]

# Create DataLoader for training, validation, and test splits
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

# Optimizer and scheduler setup
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * 3,  # 3 epochs
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(3):
    print(f"\nEpoch {epoch+1}/3")
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader)

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

    print(f"Average loss: {total_loss / len(train_dataloader):.4f}")


import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_preds, all_labels = [], []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in dataloader:
            # Move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            # Get the predicted class (most likely label)
            preds = torch.argmax(logits, dim=-1)

            # Flatten the predictions and labels to 1D arrays for comparison
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Calculate and print metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"\n📊 Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Run evaluation on the validation set
evaluate(model, val_dataloader, device)