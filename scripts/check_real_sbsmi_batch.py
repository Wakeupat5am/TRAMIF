import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.malsbslcnet import MalSBSLCNet
from scripts.sbsmi_pilot_dataset import SBSMIPilotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Folder containing the completed 10-sample pilot.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    project_root = Path(__file__).resolve().parents[1]
    mapping_path = project_root / "docs" / "class_mapping.json"

    dataset = SBSMIPilotDataset(
        run_dir=args.run_dir,
        mapping_path=mapping_path,
    )

    if len(dataset) != 10:
        raise ValueError("Expected the completed 10-sample pilot.")

    # Validate every pilot image before taking the first batch.
    for index in range(len(dataset)):
        dataset[index]

    print("Run folder:", args.run_dir.resolve())
    print("Verified pilot samples:", len(dataset))
    print("Number of classes:", dataset.num_classes)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    if tuple(images.shape) != (4, 1, 64, 64):
        raise RuntimeError("Unexpected input batch shape.")

    if images.dtype != torch.float32:
        raise RuntimeError("Images must use float32.")

    if labels.dtype != torch.int64:
        raise RuntimeError("Labels must use int64.")

    if not torch.isfinite(images).all().item():
        raise RuntimeError("Images contain non-finite values.")

    if images.min().item() < 0 or images.max().item() > 1:
        raise RuntimeError("Images must be normalized to [0, 1].")

    if labels.min().item() < 0 or labels.max().item() >= 51:
        raise RuntimeError("Labels are outside the frozen class range.")

    print("Batch shape:", tuple(images.shape))
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    print("Labels:", labels.tolist())

    for sample, label in zip(dataset.samples[:4], labels.tolist()):
        print(f"  {sample['family']} -> {label}")

    torch.manual_seed(42)
    device = torch.device("cuda")

    model = MalSBSLCNet(num_classes=dataset.num_classes).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-6,
    )
    criterion = nn.CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)

    previous_weights = model.classifier.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)

    if tuple(logits.shape) != (4, 51):
        raise RuntimeError("Unexpected model output shape.")

    loss = criterion(logits, labels)

    if not torch.isfinite(loss).item():
        raise RuntimeError("Loss is not finite.")

    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient: {name}")

        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"Non-finite gradient: {name}")

    optimizer.step()

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError(f"Non-finite parameter after update: {name}")

    if torch.equal(previous_weights, model.classifier.weight.detach()):
        raise RuntimeError("Classifier weights did not change.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Logits shape:", tuple(logits.shape))
    print("Loss before update:", loss.item())
    print("Finite gradients: PASS")
    print("Optimizer updated classifier weights: PASS")
    print("Real SBSMI batch smoke check: PASS")
    print("No checkpoint saved; this is not a classification evaluation.")


if __name__ == "__main__":
    main()