# ═══════════════════════════════════════════════════════════════════
# ResNet-TSN Model for Breathing Detection
# Temporal Segment Network with ResNet18 Backbone
# 16 Temporal Segments for Better Motion Capture
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# Cell 1 — Imports & Paths
# ═══════════════════════════════════════════════════════════════════

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import time
import gc

# Clear GPU memory
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
gc.collect()

# Paths
chest_pain_folder        = "E:/Breathing Detection/datasets/MyVideos/chest_pain/"
chest_pain_syn_folder    = "E:/Breathing Detection/datasets/synthetic/chest_pain/"
chest_col_folder         = "E:/Breathing Detection/datasets/MyVideos/combined/chest_pain_collapse/"
chest_col_syn_folder     = "E:/Breathing Detection/datasets/synthetic/combined/chest_pain_collapse/"
collapse_folder          = "E:/Breathing Detection/datasets/MyVideos/collapse/"
collapse_syn_folder      = "E:/Breathing Detection/datasets/synthetic/collapse/"
agonal_folder            = "E:/Breathing Detection/datasets/MyVideos/agonal_breathing/"
agonal_syn_folder        = "E:/Breathing Detection/datasets/synthetic/agonal_breathing/"
agonal_col_folder        = "E:/Breathing Detection/datasets/MyVideos/combined/agonal_collapse/"
agonal_col_syn_folder    = "E:/Breathing Detection/datasets/synthetic/combined/agonal_collapse/"
negative_folder          = "E:/Breathing Detection/datasets/MyVideos/negative_examples/"
mpscr_folder             = "E:/Breathing Detection/datasets/MPSC-RR/"

model_save_path          = "E:/Breathing Detection/outputs/resnet_tsn_best.pth"
checkpoint_path          = "E:/Breathing Detection/outputs/resnet_tsn_checkpoint.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True

print("Using device:", device)
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ═══════════════════════════════════════════════════════════════════
# Cell 2 — Install Missing Libraries
# ═══════════════════════════════════════════════════════════════════

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "seaborn", "matplotlib"])
print("✅ Libraries ready!")


# ═══════════════════════════════════════════════════════════════════
# Cell 3 — Dataset Class (TSN - Temporal Segment Network)
# 16 Segments for Better Temporal Representation
# ═══════════════════════════════════════════════════════════════════

class VideoDataset(Dataset):
    """
    TSN Dataset: Samples num_segments frames uniformly across the video.
    Each frame is processed independently by ResNet18.
    
    16 segments provides good temporal coverage for motion analysis.
    """
    def __init__(self, folder, label, num_segments=16, max_videos=None):
        self.folder       = folder
        self.label        = label
        self.num_segments = num_segments
        self.img_size     = 224  # ResNet18 standard
        
        videos = sorted([f for f in os.listdir(folder)
                         if f.lower().endswith((".mp4", ".mov", ".avi"))])
        if max_videos:
            videos = videos[:max_videos]
        self.videos = videos
        tag = "/".join(folder.rstrip("/").split("/")[-2:])
        print(f"  {tag:<45} {len(self.videos):>4} videos  label:{label}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        cap          = cv2.VideoCapture(os.path.join(self.folder, self.videos[idx]))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return (torch.zeros(self.num_segments, 3, self.img_size, self.img_size),
                    torch.tensor(self.label, dtype=torch.float32))

        # TSN: Divide video into num_segments equal segments
        # Sample 1 frame uniformly from each segment
        if total_frames >= self.num_segments:
            indices = np.linspace(0, total_frames - 1, self.num_segments, dtype=int)
        else:
            indices = list(range(total_frames))

        frames = []
        mean   = np.array([0.485, 0.456, 0.406])
        std    = np.array([0.229, 0.224, 0.225])

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.float32))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size)).astype(np.float32) / 255.0
            frame = (frame - mean) / std
            frames.append(frame)
        cap.release()

        # Pad if needed
        while len(frames) < self.num_segments:
            frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.float32))

        frames = np.array(frames[:self.num_segments], dtype=np.float32)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
        return frames, torch.tensor(self.label, dtype=torch.float32)


print("Loading datasets...  [N=Normal  A=Agonal  Co=Collapse  Ch=Chest]\n")

#                                               label=[N, A, Co, Ch]
# ── Normal class ─────────────────────────────────────────────────
ds_normal     = VideoDataset(mpscr_folder,          label=[1, 0, 0, 0])               #  29
ds_negative   = VideoDataset(negative_folder,       label=[1, 0, 0, 0])               # 211 ✅ NO CAP
# Normal total: 240

# ── Agonal class ─────────────────────────────────────────────────
ds_agonal     = VideoDataset(agonal_folder,         label=[0, 1, 0, 0])               #  16
ds_agonal_syn = VideoDataset(agonal_syn_folder,     label=[0, 1, 0, 0])               # 312 ✅ NO CAP
ds_agonal_col     = VideoDataset(agonal_col_folder,     label=[0, 1, 1, 0])           #   1
ds_agonal_col_syn = VideoDataset(agonal_col_syn_folder, label=[0, 1, 1, 0])           #   7
# Agonal total: 336

# ── Collapse class ───────────────────────────────────────────────
ds_collapse     = VideoDataset(collapse_folder,     label=[0, 0, 1, 0])               #  30
ds_collapse_syn = VideoDataset(collapse_syn_folder, label=[0, 0, 1, 0])               # 154 ✅ NO CAP
ds_chest_col    = VideoDataset(chest_col_folder,    label=[0, 0, 1, 1])               #   8
ds_chest_col_syn= VideoDataset(chest_col_syn_folder,label=[0, 0, 1, 1])               #  56
# Collapse total: 248

# ── Chest Pain class ─────────────────────────────────────────────
ds_chest_pain     = VideoDataset(chest_pain_folder,     label=[0, 0, 0, 1])           #  24
ds_chest_pain_syn = VideoDataset(chest_pain_syn_folder, label=[0, 0, 0, 1])           # 293 ✅ NO CAP
# Chest total: 375

combined = ConcatDataset([
    ds_normal,    ds_negative,
    ds_agonal,    ds_agonal_syn,    ds_agonal_col,     ds_agonal_col_syn,
    ds_collapse,  ds_collapse_syn,
    ds_chest_col, ds_chest_col_syn,
    ds_chest_pain,ds_chest_pain_syn,
])

total      = len(combined)
train_size = int(0.70 * total)
val_size   = int(0.15 * total)
test_size  = total - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    combined, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0, pin_memory=USE_AMP)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0, pin_memory=USE_AMP)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=0, pin_memory=USE_AMP)

print(f"\nTotal:{total}  Train:{train_size}  Val:{val_size}  Test:{test_size}")
print("\n✅ USING ALL AVAILABLE VIDEOS (1,141 total)")
print("   Temporal Segment Network: 16 segments per video")
print("   Batch size: 8")
print("   Data loading: Single-threaded (Windows compatible)")


# ═══════════════════════════════════════════════════════════════════
# Cell 4 — Model Definition (ResNet-TSN)
# ═══════════════════════════════════════════════════════════════════

class ResNet_TSN(nn.Module):
    """
    Temporal Segment Network with ResNet18 backbone.
    
    Architecture:
    - ResNet18 extracts 512-d features from each segment frame
    - Segment consensus = average pooling across all segment features
    - Final classifier maps 512-d → 4 outputs
    
    No LSTM — temporal reasoning comes from segment sampling across the video.
    """
    def __init__(self, num_segments=16, num_classes=4):
        super(ResNet_TSN, self).__init__()
        self.num_segments = num_segments

        # Pretrained ResNet18 — strip final FC layer
        backbone      = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Keep everything up to avgpool → 512-d features per frame
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Classifier applied after consensus
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W) where T = num_segments
        batch, seq, c, h, w = x.size()

        # Process all segment frames through ResNet18
        x = x.view(batch * seq, c, h, w)   # (B*T, C, H, W)
        x = self.backbone(x)                # (B*T, 512, 1, 1)
        x = x.view(batch * seq, -1)         # (B*T, 512)

        # Reshape and apply segment consensus (mean across segments)
        x = x.view(batch, seq, 512)         # (B, T, 512)
        x = x.mean(dim=1)                   # (B, 512) — consensus

        x = self.classifier(x)              # (B, 4)
        return x


model = ResNet_TSN(num_segments=16, num_classes=4).to(device)

# Freeze ResNet backbone — only classifier trains
for param in model.backbone.parameters():
    param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Model: ResNet-TSN (16 segments)")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  ResNet18 backbone: FROZEN")
print(f"  Training: Classifier only")


# ═══════════════════════════════════════════════════════════════════
# Cell 5 — Loss, Optimizer & Scheduler
# ═══════════════════════════════════════════════════════════════════

criterion  = nn.BCEWithLogitsLoss()

optimizer  = torch.optim.Adam([
    {"params": model.classifier.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
])

scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode    = "max",
    patience= 3,
    factor  = 0.3
)

scaler = torch.amp.GradScaler('cuda')

print("Loss          : BCEWithLogitsLoss")
print("Optimizer     : Adam (Classifier only)")
print("Learning Rate : 1e-4")
print("Scheduler     : ReduceLROnPlateau")
print("Precision     : Mixed AMP")


# ═══════════════════════════════════════════════════════════════════
# Cell 6 — Training Loop (20 EPOCHS, PATIENCE=7)
# ═══════════════════════════════════════════════════════════════════

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    correct    = np.zeros(4)
    n          = 0
    
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                outputs = model(frames)
                loss    = criterion(outputs, labels)
            
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            total_loss += loss.item()
            pred = (torch.sigmoid(outputs) > 0.5).float()
            for c in range(4):
                correct[c] += (pred[:, c] == labels[:, c]).sum().item()
            n += labels.size(0)
    
    return total_loss / len(loader), 100 * correct / n


# Check if resuming
checkpoint_path = "E:/Breathing Detection/outputs/resnet_tsn_checkpoint.pth"
model_save_path = "E:/Breathing Detection/outputs/resnet_tsn_best.pth"

if os.path.exists(checkpoint_path):
    print("🔄 Found checkpoint! Resuming training...\n")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_acc = ckpt["best_val_acc"]
    no_improve = ckpt["no_improve"]
    
    print(f"✅ Resumed from epoch {start_epoch}")
    print(f"   Best val acc so far: {best_val_acc:.1f}%")
    print(f"   No improve count: {no_improve}\n")
else:
    print("📁 No checkpoint found. Starting FRESH training...\n")
    
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        print(f"✅ Deleted old model\n")
    
    start_epoch = 0
    best_val_acc = 0.0
    no_improve = 0

print(f"🚀 TRAINING: ResNet-TSN | 1,141 videos | Epochs {start_epoch+1}-20 | Batch=8 | Segments=16\n")

epochs   = 20
patience = 7

for epoch in range(start_epoch, epochs):
    epoch_start = time.time()
    
    train_loss, train_accs = run_epoch(train_loader, train=True)
    val_loss,   val_accs   = run_epoch(val_loader,   train=False)
    
    train_overall = train_accs.mean()
    val_overall   = val_accs.mean()
    scheduler.step(val_overall)
    
    epoch_time = time.time() - epoch_start
    eta_remaining = epoch_time * (epochs - epoch - 1) / 60  # minutes
    
    print(f"\nEpoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s, ETA: {eta_remaining:.0f}min)")
    print(f"  Train → Loss:{train_loss:.4f} | N:{train_accs[0]:.1f}% A:{train_accs[1]:.1f}% Co:{train_accs[2]:.1f}% Ch:{train_accs[3]:.1f}% | Avg:{train_overall:.1f}%")
    print(f"  Val   → Loss:{val_loss:.4f}   | N:{val_accs[0]:.1f}%  A:{val_accs[1]:.1f}%  Co:{val_accs[2]:.1f}%  Ch:{val_accs[3]:.1f}%  | Avg:{val_overall:.1f}%")
    
    # Save checkpoint every epoch
    torch.save({
        "epoch"          : epoch,
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc"   : best_val_acc,
        "no_improve"     : no_improve,
    }, checkpoint_path)
    
    # Save best model
    if val_overall > best_val_acc:
        best_val_acc = val_overall
        no_improve   = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"  ✅ Best model saved! Val Avg: {val_overall:.1f}%")
    else:
        no_improve += 1
        print(f"  ⚠️  No improvement for {no_improve} epochs")
    
    if no_improve >= patience:
        print(f"\n  🛑 Early stopping at epoch {epoch+1}")
        break

print(f"\nTraining complete! Best val avg acc: {best_val_acc:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# Cell 7 — Test Set Evaluation
# ═══════════════════════════════════════════════════════════════════

# Load best model
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# Evaluate on test set
test_loss, test_accs = run_epoch(test_loader, train=False)
test_overall = test_accs.mean()

print("\n" + "="*65)
print("        ResNet-TSN Model — Test Results")
print("="*65)
print(f"  Normal Accuracy         {test_accs[0]:>6.1f}%")
print(f"  Agonal Accuracy         {test_accs[1]:>6.1f}%")
print(f"  Collapse Accuracy       {test_accs[2]:>6.1f}%")
print(f"  Chest Pain Accuracy     {test_accs[3]:>6.1f}%")
print("-"*65)
print(f"  Overall Test Accuracy   {test_overall:>6.1f}%")
print("="*65)


# ═══════════════════════════════════════════════════════════════════
# Cell 8 — Confusion Matrix & Classification Report
# ═══════════════════════════════════════════════════════════════════

model.eval()

all_preds  = []
all_labels = []
class_names = ["Normal", "Agonal", "Collapse", "Chest Pain"]

with torch.no_grad():
    for frames, labels in val_loader:
        frames  = frames.to(device)
        outputs = model(frames)
        
        predicted = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
        labels    = labels.cpu().numpy()
        
        for i in range(len(predicted)):
            all_preds.append(predicted[i])
            all_labels.append(labels[i])

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Per class confusion matrix
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes      = axes.flatten()

for i, class_name in enumerate(class_names):
    cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
    sns.heatmap(
        cm, annot=True, fmt="d", ax=axes[i],
        cmap="Blues",
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No",    "Actual Yes"]
    )
    axes[i].set_title(f"{class_name} Confusion Matrix")

plt.tight_layout()
plt.savefig("E:/Breathing Detection/outputs/resnet_tsn_confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved!")

# Classification report
print("\nClassification Report:")
for i, class_name in enumerate(class_names):
    print(f"\n{class_name}:")
    print(classification_report(
        all_labels[:, i],
        all_preds[:, i],
        target_names=["No", "Yes"]
    ))
