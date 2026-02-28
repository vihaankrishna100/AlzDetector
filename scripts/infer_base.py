from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from adgent.efficient_netv2 import EffNetV2Clinical
from adgent.dataset import ADNIDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = EffNetV2Clinical(num_classes=2, clin_dim=3).to(device)
state = torch.load("model.pth", map_location=device)
model.load_state_dict(state)
model.eval()

ds = ADNIDataset()
loader = DataLoader(ds, batch_size=1, shuffle=False)

y_true, y_pred, y_prob = [], [], []

for img, clin, label, sid in loader:
    img = img.to(device)  
    clin = clin.to(device)  
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)

    with torch.no_grad():
        logits, _, _ = model(img, clin)  
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = int(prob > 0.5)

    y_true.append(int(label.item()))
    y_pred.append(pred)
    y_prob.append(prob)

print("\n===== BASE MODEL METRICS =====")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("AUROC:", roc_auc_score(y_true, y_prob))
