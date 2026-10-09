from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.float_image_dataset import FloatImageDataset
from scripts.sbsmi_dataset import SBSMIDataset


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_sample(dataset, index):
    tensor, label = dataset[index]
    sample = dataset.samples[index]

    path = (
        dataset.dataset_dir
        / dataset.split
        / f"{sample['sha']}.npy"
    )
    saved = np.load(path, allow_pickle=False)

    require(tuple(tensor.shape) == (1, 64, 64), "Incorrect image shape.")
    require(tensor.dtype == torch.float32, "Incorrect image dtype.")
    require(
        label == dataset.class_to_idx[sample["family"]],
        "Incorrect label.",
    )

    # Exact equality catches accidental extra normalization or rounding.
    np.testing.assert_array_equal(tensor[0].numpy(), saved)


def main():
    reference_train = SBSMIDataset(ROOT, split="train")
    reference_validation = SBSMIDataset(ROOT, split="validation")

    for view in ("raw-byte", "entropy"):
        train = FloatImageDataset(ROOT, view=view, split="train")
        validation = FloatImageDataset(ROOT, view=view, split="validation")
        february = FloatImageDataset(
            ROOT, view=view, split="validation", month="2020-02"
        )
        march = FloatImageDataset(
            ROOT, view=view, split="validation", month="2020-03"
        )

        require(len(train) == 18061, "Unexpected train count.")
        require(len(validation) == 8263, "Unexpected validation count.")
        require(len(february) == 3825, "Unexpected February count.")
        require(len(march) == 4438, "Unexpected March count.")

        require(
            train.samples == reference_train.samples,
            "Train sample identities/order differ from SBSMI.",
        )
        require(
            validation.samples == reference_validation.samples,
            "Validation sample identities/order differ from SBSMI.",
        )
        require(
            train.class_to_idx == reference_train.class_to_idx,
            "Class mapping differs from SBSMI.",
        )

        # Check four train images and both ends of each validation month.
        for index in range(4):
            check_sample(train, index)

        for dataset in (february, march):
            check_sample(dataset, 0)
            check_sample(dataset, len(dataset) - 1)

        loader = DataLoader(
            train,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        images, labels = next(iter(loader))

        require(
            tuple(images.shape) == (4, 1, 64, 64),
            "Incorrect batch shape.",
        )
        require(images.dtype == torch.float32, "Incorrect batch dtype.")
        require(labels.dtype == torch.int64, "Incorrect label dtype.")

        expected_labels = [
            train.class_to_idx[row["family"]]
            for row in train.samples[:4]
        ]
        require(labels.tolist() == expected_labels, "Batch labels differ.")

        print(f"\nView: {view}")
        print("Train samples:", len(train))
        print("Validation samples:", len(validation))
        print("February samples:", len(february))
        print("March samples:", len(march))
        print("Classes:", train.num_classes)
        print("Manifest alignment with SBSMI: PASS")
        print("Sampled NPY-to-tensor equality: PASS (8 images)")
        print("Batch shape:", tuple(images.shape))
        print("Image dtype:", images.dtype)
        print("Label dtype:", labels.dtype)
        print("Labels:", labels.tolist())

    print("\nRaw-byte/entropy dataset loading smoke check: PASS")
    print("No training or model evaluation performed.")
    print("Dataset files were not changed.")


if __name__ == "__main__":
    main()