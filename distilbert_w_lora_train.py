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
import time


# Load pre-trained DistillBert model
model = AutoModelForSequenceClassification.from_pretrained("./distilbert-base-uncased", num_labels=2)

# Freeze all the parameters of DistillBert
for param in model.parameters():
    param.requires_grad = False

# # Show the architecture of DistillBert
# print(model)

# Apply LoRA
lora_rank = 8
lora_alpha = 16
lora = partial(LinearWithLoRA, rank=lora_rank, alpha=lora_alpha)
for name, module in model.named_modules():
    if isinstance(module, nn.Module):
        # Apply LoRA to attention layers
        if hasattr(module, 'attention'):
            if hasattr(module.attention, 'q_lin'):
                module.attention.q_lin = lora(module.attention.q_lin)
            if hasattr(module.attention, 'k_lin'):
                module.attention.k_lin = lora(module.attention.k_lin)
            if hasattr(module.attention, 'v_lin'):
                module.attention.v_lin = lora(module.attention.v_lin)
            if hasattr(module.attention, 'out_lin'):
                module.attention.out_lin = lora(module.attention.out_lin)

        # Apply LoRA to MLP layers 
        if hasattr(module, 'ffn'):
            if hasattr(module.ffn, 'lin1'):
                module.ffn.lin1 = lora(module.ffn.lin1)
            if hasattr(module.ffn, 'lin2'):
                module.ffn.lin2 = lora(module.ffn.lin2)

        # Apply LoRA to classifier layers
        if hasattr(module, 'pre_classifier'):
            module.pre_classifier = lora(module.pre_classifier)
        if hasattr(module, 'classifier'):
            module.classifier = lora(module.classifier)

# # Show the architecture of DistillBert after replacement (Linear -> LinearWithLoRA)
# print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable parameters:", count_parameters(model))

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
print("Tokenizer input max length:", tokenizer.model_max_length)
print("Tokenizer vocabulary size:", tokenizer.vocab_size)

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

# Training pharse
epochs = 10
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.to(device)
best_val_loss = float("inf") 
start_time = time.time()
for epoch in tqdm(range(epochs)):
    # Training
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for features, attention_mask, targets in train_loader:
        features = features.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        # Forward
        output = model(input_ids=features, attention_mask=attention_mask)
        loss = F.cross_entropy(output.logits, targets)
        train_loss += loss.item()
        # Calculate accuray
        preds = output.logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    train_accuracy = correct / total
    print(f"[Info] Epoch {epoch+1} Training Loss: {train_loss:.4f} Training Accuracy: {train_accuracy:.4f} \n")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, attention_mask, targets in val_loader:
            features = features.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            # Forward 
            output = model(input_ids=features, attention_mask=attention_mask)
            loss = F.cross_entropy(output.logits, targets)
            val_loss += loss.item()
            # Calculate accuray
            preds = output.logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    val_loss /= len(val_loader)
    val_accuracy = correct / total
    print(f"[Info] Epoch {epoch+1} Validation Loss: {val_loss:.4f} Validation Accuracy: {val_accuracy:.4f}, Training time: {time.time() - start_time}s \n")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "./checkpoints/best_distillbert_lora_model.pth")
        print(f"[Info] Best model saved at Epoch {epoch+1} \n")

