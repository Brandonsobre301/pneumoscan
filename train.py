"""
Training script for PneumoScan ResNet-18 model.
Uses the Kermany chest X-ray dataset with transfer learning.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(101010)
np.random.seed(101010)
random.seed(101010)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT  = r"C:\Users\brand\Downloads\workspace(1)\workspace\data\chestxrays"
TRAIN_DIR  = DATA_ROOT + r"\train"
VAL_DIR    = DATA_ROOT + r"\val"
TEST_DIR   = DATA_ROOT + r"\test"
SAVE_PATH  = "resnet18_pneumonia.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR         = 1e-3

# ── Transforms ───────────────────────────────────────────────────────────────
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# ── Datasets & Loaders ───────────────────────────────────────────────────────
train_ds = ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds   = ImageFolder(VAL_DIR,   transform=eval_transform)
test_ds  = ImageFolder(TEST_DIR,  transform=eval_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Classes: {train_ds.classes}")

# ── Model ────────────────────────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers, then unfreeze layer4 + fc for fine-tuning
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 1)  # fc is always trainable (new layer)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        running_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        total   += inputs.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            preds = (torch.sigmoid(model(inputs)) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total   += inputs.size(0)

    val_acc = val_correct / val_total
    scheduler.step()

    print(f"Epoch [{epoch}/{NUM_EPOCHS}]  train_loss: {train_loss:.4f}  "
          f"train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}")

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  --> Saved best model (val_acc={val_acc:.4f})")

# ── Final test evaluation ─────────────────────────────────────────────────────
print("\n--- Final Test Evaluation ---")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        preds  = (torch.sigmoid(model(inputs)) > 0.5).cpu().long().squeeze(1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

all_preds  = torch.tensor(all_preds)
all_labels = torch.tensor(all_labels)

test_acc = (all_preds == all_labels).float().mean().item()

# F1 (binary, PNEUMONIA=1)
tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test F1-Score : {f1:.4f}")
print(f"\nModel saved to: {SAVE_PATH}")
