import torch
from torch import nn

from models.malsbslcnet import MalSBSLCNet


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the environment.")

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    # Nine classes match the supplementary table's classifier size.
    reference_model = MalSBSLCNet(num_classes=9)
    reference_count = count_parameters(reference_model)
    del reference_model

    model = MalSBSLCNet(num_classes=51).to(device)
    project_count = count_parameters(model)

    print("Parameters with 9 classes:", reference_count)
    print("Parameters with 51 classes:", project_count)

    # Counts calculated from the supplementary layer table.
    if reference_count != 54409:
        raise RuntimeError("Parameter count differs from the 9-class table.")

    if project_count != 140467:
        raise RuntimeError("Unexpected parameter count for 51 classes.")

    # Inspect the shapes without updating BatchNorm statistics.
    model.eval()
    x = torch.rand(2, 1, 64, 64, device=device)

    stages = [
        ("stem", model.stem, (2, 16, 64, 64)),
        ("block1", model.blocks[0], (2, 32, 32, 32)),
        ("block2", model.blocks[1], (2, 32, 32, 32)),
        ("block3", model.blocks[2], (2, 64, 16, 16)),
        ("block4", model.blocks[3], (2, 64, 16, 16)),
        ("block5", model.blocks[4], (2, 128, 8, 8)),
        ("block6", model.blocks[5], (2, 128, 8, 8)),
        ("pool", model.pool, (2, 128, 4, 4)),
        ("flatten", model.flatten, (2, 2048)),
        ("head_norm", model.head_norm, (2, 2048)),
        ("dropout", model.dropout, (2, 2048)),
        ("classifier", model.classifier, (2, 51)),
    ]

    with torch.no_grad():
        features = x

        for name, layer, expected_shape in stages:
            features = layer(features)
            actual_shape = tuple(features.shape)
            print(f"{name}: {actual_shape}")

            if actual_shape != expected_shape:
                raise RuntimeError(f"Unexpected shape at {name}.")

    # Confirm that loss and gradients work in training mode.
    model.train()
    model.zero_grad(set_to_none=True)

    labels = torch.tensor([0, 50], dtype=torch.long, device=device)
    logits = model(x)

    if tuple(logits.shape) != (2, 51):
        raise RuntimeError("Unexpected training output shape.")

    loss = nn.CrossEntropyLoss()(logits, labels)

    if not torch.isfinite(loss).item():
        raise RuntimeError("Loss is not finite.")

    loss.backward()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient: {name}")

        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"Non-finite gradient: {name}")

    print("Forward/backward on GPU: PASS")

    # In evaluation mode, predicting a single image is supported.
    model.eval()

    with torch.no_grad():
        single_logits = model(x[:1])
        probabilities = torch.softmax(single_logits, dim=1)

    if tuple(single_logits.shape) != (1, 51):
        raise RuntimeError("Unexpected single-image output shape.")

    if not torch.isfinite(probabilities).all().item():
        raise RuntimeError("Invalid probabilities.")

    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(1, device=device),
    )

    print("Single-image evaluation: PASS")
    print("Architecture smoke check: PASS")


if __name__ == "__main__":
    main()