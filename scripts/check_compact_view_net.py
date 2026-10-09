from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.compact_view_net import CompactViewNet
from models.malsbslcnet import MalSBSLCNet
from scripts.float_image_dataset import FloatImageDataset


ROOT = Path(__file__).resolve().parents[1]
PARAMETER_TARGET = 300000


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_view(view, device):
    dataset = FloatImageDataset(ROOT, view=view, split="train")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Each view receives a newly initialized, independent model.
    model = CompactViewNet(num_classes=dataset.num_classes).to(device)

    model.eval()
    with torch.no_grad():
        features = model.forward_features(images)
        logits = model(images)

    require(
        tuple(features.shape) == (4, 512),
        f"{view}: incorrect feature shape.",
    )
    require(
        tuple(logits.shape) == (4, 51),
        f"{view}: incorrect output shape.",
    )
    require(
        torch.isfinite(logits).all().item(),
        f"{view}: non-finite output.",
    )

    model.train()
    model.zero_grad(set_to_none=True)

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)

    require(torch.isfinite(loss).item(), f"{view}: non-finite loss.")
    loss.backward()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        require(
            parameter.grad is not None,
            f"{view}: missing gradient for {name}.",
        )
        require(
            torch.isfinite(parameter.grad).all().item(),
            f"{view}: non-finite gradient for {name}.",
        )

    require(
        model.classifier.weight.grad.abs().sum().item() > 0,
        f"{view}: classifier gradient is entirely zero.",
    )

    model.eval()
    with torch.no_grad():
        single_logits = model(images[:1])

    require(
        tuple(single_logits.shape) == (1, 51),
        f"{view}: incorrect single-image output shape.",
    )
    require(
        torch.isfinite(single_logits).all().item(),
        f"{view}: non-finite single-image output.",
    )

    print(f"\nView: {view}")
    print("Batch shape:", tuple(images.shape))
    print("Feature shape:", tuple(features.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Labels:", labels.tolist())
    print("Initial batch loss:", round(loss.item(), 6))
    print("Finite gradients: PASS")
    print("Nonzero classifier gradient: PASS")
    print("Single-image inference: PASS")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    original = MalSBSLCNet(num_classes=51)
    compact = CompactViewNet(num_classes=51)

    original_count = count_parameters(original)
    compact_count = count_parameters(compact)
    branch_total = original_count + 2 * compact_count

    require(original_count == 140467, "Unexpected SBSMI parameter count.")
    require(compact_count == 59059, "Unexpected compact parameter count.")
    require(
        branch_total <= PARAMETER_TARGET,
        "Three-branch parameter count exceeds the design target.",
    )

    del original, compact

    print("GPU:", torch.cuda.get_device_name(0))
    print("SBSMI parameters:", original_count)
    print("Parameters per compact branch:", compact_count)
    print("Three-branch parameters:", branch_total)
    print("Design target:", PARAMETER_TARGET)
    print("Branch parameter budget: PASS")
    print("Fusion components are not included in this count.")

    device = torch.device("cuda")
    for view in ("raw-byte", "entropy"):
        check_view(view, device)

    print("\nCompact branch smoke check: PASS")
    print("Only four training samples per view were used.")
    print("No optimizer steps. No checkpoint saved.")
    print("No validation or future-test data used.")


if __name__ == "__main__":
    main()