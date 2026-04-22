import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform) # 50,000 images
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform) # 10,1000 images

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.gate_scores, 2.0)  
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sparsity_loss(self):
        loss = 0.0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                loss = loss + module.get_gates().sum()
        return loss
    
def train_one_epoch(model, loader, optimizer, criterion, device, lambda_):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        cls_loss = criterion(outputs, labels)
        sparse_loss = model.sparsity_loss()
        loss = cls_loss + lambda_ * sparse_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = 100.0 * total_correct / total_samples
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = 100.0 * total_correct / total_samples
    return avg_loss, acc

def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gates()
                total += gates.numel()
                pruned += (gates < threshold).sum().item()

    return 100.0 * pruned / total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SelfPruningNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

lambda_ = 1e-5
epochs = 5

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, lambda_)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    sparsity = compute_sparsity(model)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")