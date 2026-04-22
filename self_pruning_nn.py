import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform) # 50,000 images
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform) # 10,000 images

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

    def sparsity_loss(self, mode="l1"):
        loss = 0.0

        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gates()

                if mode == "l1":
                    # normalized L1 (better than your current)
                    loss = loss + gates.sum() / gates.numel()

                elif mode == "binarize":
                    # encourages gates -> 0 or 1
                    loss = loss + (gates * (1 - gates)).sum() / gates.numel()

        return loss
    
def train_one_epoch(model, loader, optimizer, criterion, device, lambda_):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_sparse_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        cls_loss = criterion(outputs, labels)
        sparse_loss = model.sparsity_loss(mode="binarize")
        loss = cls_loss + lambda_ * sparse_loss

        loss.backward()

        if batch_idx % 100 == 0:
            # Safe gradient printing for fc1
            if model.fc1.gate_scores.grad is not None:
                print(f"fc1 gate grad mean: {model.fc1.gate_scores.grad.abs().mean().item():.6f}")
            else:
                print("fc1 gate grad mean: None")
                
            # Safe gradient printing for fc2
            if model.fc2.gate_scores.grad is not None:
                print(f"fc2 gate grad mean: {model.fc2.gate_scores.grad.abs().mean().item():.6f}")
            else:
                print("fc2 gate grad mean: None")
            
            # Safe gradient printing for fc3
            if model.fc3.gate_scores.grad is not None:
                print(f"fc3 gate grad mean: {model.fc3.gate_scores.grad.abs().mean().item():.6f}")
            else:
                print("fc3 gate grad mean: None")
        
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_sparse_loss += sparse_loss.item()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Optional batch-level debug print every 100 batches
        if batch_idx % 100 == 0:
            print(
                f"Batch {batch_idx}/{len(loader)} | "
                f"Cls Loss: {cls_loss.item():.4f} | "
                f"Sparse Loss: {sparse_loss.item():.2f} | "
                f"Weighted Sparse: {(lambda_ * sparse_loss).item():.4f}"
            )

    avg_loss = total_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_sparse_loss = total_sparse_loss / len(loader)
    acc = 100.0 * total_correct / total_samples

    return avg_loss, avg_cls_loss, avg_sparse_loss, acc

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

def get_gate_statistics(model):
    all_gates = []

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gates().view(-1)
                all_gates.append(gates)

    all_gates = torch.cat(all_gates)

    stats = {
        "mean_gate": all_gates.mean().item(),
        "min_gate": all_gates.min().item(),
        "max_gate": all_gates.max().item(),
        "below_0.5": (all_gates < 0.5).float().mean().item() * 100,
        "below_0.1": (all_gates < 0.1).float().mean().item() * 100,
        "below_0.01": (all_gates < 0.01).float().mean().item() * 100,
    }

    return stats


def print_layerwise_gate_stats(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gates()
                print(
                    f"{name}: mean={gates.mean().item():.4f}, "
                    f"min={gates.min().item():.4f}, "
                    f"max={gates.max().item():.4f}, "
                    f"<0.1={(gates < 0.1).float().mean().item() * 100:.2f}%"
                )

def print_final_network_report(lambda_, final_test_acc, final_sparsity, stats):
    print("\n" + "=" * 70)
    print("FINAL STATE OF THE NETWORK")
    print("=" * 70)
    print(f"Lambda (λ):              {lambda_}")
    print(f"Final Test Accuracy (%): {final_test_acc:.2f}")
    print(f"Final Sparsity (%):      {final_sparsity:.2f}")
    print(f"Mean Gate Value:         {stats['mean_gate']:.4f}")
    print(f"Minimum Gate Value:      {stats['min_gate']:.6f}")
    print(f"Maximum Gate Value:      {stats['max_gate']:.4f}")
    print(f"Gates < 0.5 (%):         {stats['below_0.5']:.2f}")
    print(f"Gates < 0.1 (%):         {stats['below_0.1']:.2f}")
    print(f"Gates < 0.01 (%):        {stats['below_0.01']:.2f}")
    print("=" * 70)


def print_results_table(results):
    print("\n" + "=" * 95)
    print(f"{'Lambda (λ)':<15}{'Final Test Accuracy (%)':<28}{'Final Sparsity (%)':<22}{'Observation':<30}")
    print("=" * 95)

    for row in results:
        print(
            f"{row['lambda']:<15}"
            f"{row['final_test_accuracy']:<28.2f}"
            f"{row['final_sparsity']:<22.2f}"
            f"{row['observation']:<30}"
        )

    print("=" * 95)

def save_gate_histogram(model, lambda_):
    all_gates = []

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().view(-1).cpu())

    all_gates = torch.cat(all_gates).numpy()

    plt.figure(figsize=(8, 6))
    plt.hist(all_gates, bins=50)
    plt.xlabel("Gate Value")
    plt.ylabel("Number of Gates")
    plt.title(f"Final Gate Distribution (lambda={lambda_})")
    plt.grid(True)

    filename = f"gates_lambda_{str(lambda_).replace('.', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved gate histogram to: {filename}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

lambda_values = [1e-4]
results = []
epochs = 10

for lambda_ in lambda_values:
    print("\n" + "#" * 70)
    print(f"RUNNING EXPERIMENT FOR LAMBDA = {lambda_}")
    print("#" * 70)

    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_cls_loss, train_sparse_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, lambda_
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        sparsity = compute_sparsity(model)
        stats = get_gate_statistics(model)

        print(f"Train Total Loss: {train_loss:.4f}")
        print(f"Train Cls Loss:   {train_cls_loss:.4f}")
        print(f"Train Sparse Loss:{train_sparse_loss:.2f}")
        print(f"Train Acc:        {train_acc:.2f}%")
        print(f"Test Loss:        {test_loss:.4f}")
        print(f"Test Acc:         {test_acc:.2f}%")
        print(f"Sparsity (<1e-2): {sparsity:.2f}%")
        print(
            f"Gates -> mean: {stats['mean_gate']:.4f}, "
            f"min: {stats['min_gate']:.6f}, "
            f"max: {stats['max_gate']:.4f}"
        )
        print(
            f"% gates < 0.5: {stats['below_0.5']:.2f}% | "
            f"< 0.1: {stats['below_0.1']:.2f}% | "
            f"< 0.01: {stats['below_0.01']:.2f}%"
        )

        print_layerwise_gate_stats(model)

    # Final state after training finishes for this lambda
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    final_sparsity = compute_sparsity(model)
    final_stats = get_gate_statistics(model)

    print_final_network_report(lambda_, final_test_acc, final_sparsity, final_stats)
    save_gate_histogram(model, lambda_)

    results.append({
        "lambda": lambda_,
        "final_test_accuracy": final_test_acc,
        "final_sparsity": final_sparsity,
        "observation": "-"
    })

print_results_table(results)