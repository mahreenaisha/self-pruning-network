"""
Self-Pruning Neural Network using Learnable Gates
Train a multi-layer perceptron on CIFAR-10 with a learnable gating mechanism.
Each weight has an associated gate that can be driven to zero, effectively pruning the connection.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import warnings
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================================
# Data Loading
# ============================================================================

# Transform to convert PIL images to PyTorch tensors (range [0,1])
transform = transforms.ToTensor()

# CIFAR-10: 50k training images, 10k test images, each 32x32 RGB
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Data loaders: batch size 128, shuffle training set for better generalization
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# ============================================================================
# Prunable Linear Layer (with Learnable Gates)
# ============================================================================

class PrunableLinear(nn.Module):
    """
    Linear layer where each weight is multiplied by a learnable gate (0..1).
    The gate is obtained by applying sigmoid to a trainable parameter `gate_scores`.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features:  Number of input features
            out_features: Number of output features
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight matrix (out_features x in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # Standard bias vector
        self.bias = nn.Parameter(torch.empty(out_features))

        # Learnable gate scores: one scalar per weight
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights (Kaiming uniform), gate scores (constant 2.0), and bias."""
        # Kaiming uniform initialization for ReLU-based layers
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Gate scores initialised to 2.0 -> sigmoid(2.0) ≈ 0.88 (gates start mostly open)
        nn.init.constant_(self.gate_scores, 2.0)
        # Bias initialization following PyTorch Linear default
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply gating to weights, then standard linear transformation.
        Args:
            x: Input tensor of shape (batch_size, in_features)
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Convert gate scores to probabilities in (0,1)
        gates = torch.sigmoid(self.gate_scores)          # shape: (out_features, in_features)
        # Prune weights by element-wise multiplication
        pruned_weight = self.weight * gates              # same shape as weight
        # Standard linear operation with pruned weights and original bias
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (after sigmoid) for monitoring / sparsity loss."""
        return torch.sigmoid(self.gate_scores)


# ============================================================================
# Complete Self-Pruning Network (MLP with three PrunableLinear layers)
# ============================================================================

class SelfPruningNet(nn.Module):
    """Multi-layer perceptron with three prunable linear layers."""
    def __init__(self):
        super().__init__()
        # Input: 32*32*3 = 3072
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)          # Output: 10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: flatten input, apply two ReLU layers, then final linear.
        Args:
            x: Input image batch of shape (batch_size, 3, 32, 32)
        Returns:
            Logits of shape (batch_size, 10)
        """
        x = x.view(x.size(0), -1)      # flatten to (batch_size, 3072)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)                # no activation on final layer (CrossEntropyLoss expects logits)
        return x

    def sparsity_loss(self) -> torch.Tensor:
        """
        Compute L1 penalty on gates: sum of all gate values across all PrunableLinear layers.
        Returns:
            Scalar tensor: total sum of gate values.
        """
        loss = 0.0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                loss = loss + module.get_gates().sum()
        return loss


# ============================================================================
# Training and Evaluation Helpers
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, lambda_, debug=False, debug_every=100):
    """
    Train the model for one epoch.
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: 'cuda' or 'cpu'
        lambda_: Weight for sparsity loss
        debug: If True, print detailed batch logs every debug_every batches
        debug_every: Frequency of debug logging (in batches)
    Returns:
        avg_loss: Average total loss over the epoch
        avg_cls_loss: Average classification loss
        avg_sparse_loss: Average sparsity loss (before lambda weighting)
        acc: Training accuracy (%)
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_sparse_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to device (GPU/CPU)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        cls_loss = criterion(outputs, labels)
        sparse_loss = model.sparsity_loss()
        loss = cls_loss + lambda_ * sparse_loss

        # Backward pass
        loss.backward()

        # Optional debug logging
        if debug and batch_idx % debug_every == 0:
            print_detailed_batch_logs(
                model=model,
                batch_idx=batch_idx,
                total_batches=len(loader),
                cls_loss=cls_loss,
                sparse_loss=sparse_loss,
                lambda_=lambda_
            )

        optimizer.step()

        # Accumulate statistics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_sparse_loss += sparse_loss.item()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    # Compute averages
    avg_loss = total_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_sparse_loss = total_sparse_loss / len(loader)
    acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_cls_loss, avg_sparse_loss, acc


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a dataset (no gradient computation).
    Args:
        model: PyTorch model
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: 'cuda' or 'cpu'
    Returns:
        avg_loss: Average loss over the dataset
        acc: Accuracy (%)
    """
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


# ============================================================================
# Sparsity and Gate Statistics
# ============================================================================

def compute_sparsity(model, threshold=1e-2):
    """
    Compute the percentage of gates below a given threshold.
    Args:
        model: PyTorch model containing PrunableLinear layers
        threshold: Value in [0,1] below which a gate is considered "pruned"
    Returns:
        float: Sparsity percentage (0-100)
    """
    # Input validation
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"Threshold must be a number, got {type(threshold).__name__}")
    if threshold < 0 or threshold > 1:
        raise ValueError(f"Threshold must be between 0 and 1 (inclusive), got {threshold}")

    # Warn about extreme thresholds
    if threshold == 0:
        warnings.warn("Threshold is 0 - only gates exactly 0 will be counted as pruned. "
                      "Consider using a small positive threshold like 0.01 for meaningful pruning metrics.",
                      UserWarning)
    elif threshold == 1:
        warnings.warn("Threshold is 1 - all gates will be counted as pruned since gates are in [0,1]. "
                      "Consider using a smaller threshold like 0.1 for meaningful pruning metrics.",
                      UserWarning)

    total = 0
    pruned = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gates()          # values in (0,1)
                total += gates.numel()
                pruned += (gates < threshold).sum().item()

    return 100.0 * pruned / total if total > 0 else 0.0


def get_gate_statistics(model):
    """
    Aggregate global gate statistics across all prunable layers.
    Args:
        model: PyTorch model
    Returns:
        dict: mean, min, max gate values, and percentages below 0.5, 0.1, 0.01
    """
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
    """Print per-layer gate statistics (mean, min, max, % below 0.1)."""
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


# ============================================================================
# Reporting and Visualization
# ============================================================================

def print_final_network_report(lambda_, final_test_acc, final_sparsity, stats):
    """Print a nicely formatted summary of final network state."""
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
    """Print a table comparing results across different lambda experiments."""
    print("\n" + "=" * 95)
    print(f"{'Lambda (λ)':<15}{'Final Test Accuracy (%)':<28}{'Final Sparsity (%)':<22}")
    print("=" * 95)
    for row in results:
        print(
            f"{row['lambda']:<15}"
            f"{row['final_test_accuracy']:<28.2f}"
            f"{row['final_sparsity']:<22.2f}"
        )
    print("=" * 95)


def print_detailed_batch_logs(model, batch_idx, total_batches, cls_loss, sparse_loss, lambda_):
    """Print detailed batch-level logs including gradient means for debugging."""
    # Gate gradient means (absolute) for each prunable layer
    if model.fc1.gate_scores.grad is not None:
        print(f"fc1 gate grad mean: {model.fc1.gate_scores.grad.abs().mean().item():.6f}")
    else:
        print("fc1 gate grad mean: None")

    if model.fc2.gate_scores.grad is not None:
        print(f"fc2 gate grad mean: {model.fc2.gate_scores.grad.abs().mean().item():.6f}")
    else:
        print("fc2 gate grad mean: None")

    if model.fc3.gate_scores.grad is not None:
        print(f"fc3 gate grad mean: {model.fc3.gate_scores.grad.abs().mean().item():.6f}")
    else:
        print("fc3 gate grad mean: None")

    print(
        f"Batch {batch_idx}/{total_batches} | "
        f"Cls Loss: {cls_loss.item():.4f} | "
        f"Sparse Loss: {sparse_loss.item():.2f} | "
        f"Weighted Sparse: {(lambda_ * sparse_loss).item():.4f}"
    )


def save_gate_histogram(model, lambda_):
    """
    Plot a histogram of all gate values and save it to the ./plots/ directory.
    Args:
        model: PyTorch model
        lambda_: The lambda value used for this experiment (used in filename and title)
    Returns:
        Path object where the image was saved
    """
    # Collect all gate values
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().view(-1).cpu())

    all_gates = torch.cat(all_gates).numpy()

    # Create figure
    plt.figure(figsize=(8, 6))
    plt.hist(all_gates, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel("Gate Value")
    plt.ylabel("Number of Gates")
    plt.title(f"Final Gate Distribution (λ={lambda_})")
    plt.grid(True, alpha=0.3)

    # Ensure plots directory exists
    plots_dir = Path("./plots")
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Generate safe filename (replace decimal dot with underscore)
    lambda_str = f"{lambda_:.6f}".replace('.', '_')
    filename = plots_dir / f"gates_lambda_{lambda_str}.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gate histogram saved to: {filename}")
    return filename


# ============================================================================
# Main Execution (Experiments)
# ============================================================================

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # Hyperparameters
    lambda_values = [1e-6, 5e-6, 1e-5, 1e-4, 5e-4]   # Different sparsity strengths
    results = []
    epochs = 30
    DEBUG_LOGGING = False          # Set to True for detailed batch logs
    DEBUG_EVERY = 100

    # Run experiment for each lambda value
    for lambda_ in lambda_values:
        print("\n" + "#" * 70)
        print(f"RUNNING EXPERIMENT FOR LAMBDA = {lambda_}")
        print("#" * 70)

        # Create fresh model and optimizer for each lambda
        model = SelfPruningNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Train one epoch
            train_loss, train_cls_loss, train_sparse_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                lambda_, debug=DEBUG_LOGGING, debug_every=DEBUG_EVERY
            )

            # Evaluate on test set
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            # Compute sparsity at different thresholds
            sparsity_01 = compute_sparsity(model, threshold=0.1)
            sparsity_005 = compute_sparsity(model, threshold=0.05)
            sparsity_002 = compute_sparsity(model, threshold=0.02)
            sparsity_001 = compute_sparsity(model, threshold=0.01)

            # Get gate statistics
            stats = get_gate_statistics(model)

            # Print basic epoch summary
            print(f"Train Total Loss: {train_loss:.4f}")
            print(f"Train Acc:        {train_acc:.2f}%")
            print(f"Test Loss:        {test_loss:.4f}")
            print(f"Test Acc:         {test_acc:.2f}%")
            print(f"Sparsity (<0.01): {sparsity_001:.2f}%")

            # Optional detailed logs (only if DEBUG_LOGGING is True)
            if DEBUG_LOGGING:
                print(f"Sparsity (<0.1):  {sparsity_01:.2f}%")
                print(f"Sparsity (<0.05): {sparsity_005:.2f}%")
                print(f"Sparsity (<0.02): {sparsity_002:.2f}%")
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

        # After all epochs, compute final metrics and save results
        final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)

        final_sparsity_01 = compute_sparsity(model, threshold=0.1)
        final_sparsity_005 = compute_sparsity(model, threshold=0.05)
        final_sparsity_002 = compute_sparsity(model, threshold=0.02)
        final_sparsity_001 = compute_sparsity(model, threshold=0.01)

        final_stats = get_gate_statistics(model)

        print_final_network_report(lambda_, final_test_acc, final_sparsity_001, final_stats)

        print(f"Final Sparsity (<0.1):  {final_sparsity_01:.2f}%")
        print(f"Final Sparsity (<0.05): {final_sparsity_005:.2f}%")
        print(f"Final Sparsity (<0.02): {final_sparsity_002:.2f}%")
        print(f"Final Sparsity (<0.01): {final_sparsity_001:.2f}%")

        # Save gate histogram
        save_gate_histogram(model, lambda_)

        # Store results for later comparison
        results.append({
            "lambda": lambda_,
            "final_test_accuracy": final_test_acc,
            "final_sparsity": final_sparsity_001
        })

    # Print final comparison table across all lambda values
    print_results_table(results)