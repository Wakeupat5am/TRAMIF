from torch import nn

from models.malsbslcnet import BaseBlock, conv_bn


class CompactViewNet(nn.Module):
    """TRAMIF compact classifier for raw-byte or entropy images.

    Reuses the convolutional block structure of the existing MalSBSLCNet.
    Changes the final pooling grid to 2x2 and the head width to 512.

    This is a project-specific variant, not the original paper model.
    Create separate instances for the two views.
    """

    def __init__(self, num_classes=51):
        super().__init__()

        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2.")

        self.num_classes = num_classes
        self.feature_dim = 512

        self.stem = conv_bn(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
        )

        self.blocks = nn.Sequential(
            BaseBlock(16, 32, stride=2),
            BaseBlock(32, 32, stride=1),
            BaseBlock(32, 64, stride=2),
            BaseBlock(64, 64, stride=1),
            BaseBlock(64, 128, stride=2),
            BaseBlock(128, 128, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten(start_dim=1)
        self.head_norm = nn.BatchNorm1d(self.feature_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 64, 64):
            raise ValueError("Expected input shape [batch, 1, 64, 64].")

        if not x.is_floating_point():
            raise ValueError("Input must be a floating-point tensor.")

        if self.training and x.shape[0] < 2:
            raise ValueError(
                "Training requires at least two samples "
                "because the head uses BatchNorm1d."
            )

        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head_norm(x)

    def forward(self, x):
        features = self.forward_features(x)
        features = self.dropout(features)

        # Return class scores (logits), not probabilities.
        return self.classifier(features)