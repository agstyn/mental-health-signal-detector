import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/mental_health_clean.csv")

# Keep only required columns
df = df[["clean_text", "target"]]

# Remove missing text rows
df = df.dropna(subset=["clean_text"])

# Ensure text is string
df["clean_text"] = df["clean_text"].astype(str)

print("Dataset loaded:", df.shape)

# Train / Validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["clean_text"],
    df["target"],
    test_size=0.2,
    random_state=42
)

# Convert to list of strings
train_texts = train_texts.astype(str).tolist()
val_texts = val_texts.astype(str).tolist()

print("Loading tokenizer...")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

print("Tokenizing text...")

# Tokenize
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": list(train_labels)
})

val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": list(val_labels)
})

print("Loading model...")

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5
)

# Training configuration
training_args = TrainingArguments(
    output_dir="models/results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="models/logs",
    logging_steps=50
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Starting training...")

# Train model
trainer.train()

print("Saving model...")

# Save trained model
model.save_pretrained("models/distilbert_mental_health")
tokenizer.save_pretrained("models/distilbert_mental_health")

print("Training complete. Model saved in models/distilbert_mental_health")