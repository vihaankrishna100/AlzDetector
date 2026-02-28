import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from efficient_netv2 import EffNetV2Clinical
from dataset import ADNIDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

dataset = ADNIDataset()
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = EffNetV2Clinical(num_classes=2, clin_dim=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(3):   
    model.train()
    running_loss = 0

    for images, clin, labels, subject_id in train_loader:
        images = images.to(device)
        clin = clin.to(device)
        labels = labels.to(device)
        
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        optimizer.zero_grad()
        outputs, _, _ = model(images, clin)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
