import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import torch
from functools import partial
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from IMDBDataset import IMDBDataset
from LoRA import LinearWithLoRA
from datetime import datetime

# Load pre-trained DistillBert model
model = AutoModelForSequenceClassification.from_pretrained("./distilbert-base-uncased", num_labels=2)

# Detect available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained weights
model.load_state_dict(torch.load("./checkpoints/best_distillbert_model.pth", weights_only=True))
model.to(device)

# Prepare imdb datasets
imdb_dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join("data", "train.csv"),
        "validation": os.path.join("data", "val.csv"),
        "test": os.path.join("data", "test.csv"),
    },
)

# Tokenize text
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)

imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up DataLoader
train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")
train_loader = DataLoader(dataset=train_dataset, batch_size=12, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=12, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=12, num_workers=4)

# Test phase
def test(model, dataloaders=test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, attention_mask, targets in dataloaders:
            features = features.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            # Forward 
            output = model(input_ids=features, attention_mask=attention_mask)
            loss = F.cross_entropy(output.logits, targets)
            test_loss += loss.item()
            # Calculate accuracy
            preds = output.logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    test_loss /= len(dataloaders)
    test_accuracy = correct / total
    return test_loss, test_accuracy

train_loss, train_acc = test(model, train_loader)
val_loss, val_acc = test(model, val_loader)
test_loss, test_acc = test(model, test_loader)

with open("results.txt", "a") as f:
    f.write(f"{datetime.now().strftime("%Y-%m-%d %H")}"+"\n")
    f.write(f"DistillBert without LoRA \n")
    f.write(f"Train Accuracy: {train_acc*100}%, Train Loss: {train_loss}"+"\n")
    f.write(f"Val Accuracy: {val_acc*100}%, Val Loss: {val_loss}"+"\n")
    f.write(f"Test Accuracy: {test_acc*100}%, Test Loss: {test_loss}"+"\n")