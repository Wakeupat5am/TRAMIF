import torch
from torch import nn


def conv_bn(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    groups=1,
    use_relu=True,
):
    """Build convolution + BatchNorm, with an optional ReLU."""

    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]

    if use_relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class BaseBlock(nn.Module):
    """Two-branch block with channel concatenation and shuffle."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2.")

        if out_channels <= 0 or out_channels % 2 != 0:
            raise ValueError("out_channels must be positive and even.")

        if stride == 1 and in_channels != out_channels:
            raise ValueError(
                "A stride-1 block must preserve the channel count."
            )

        self.stride = stride
        branch_channels = out_channels // 2

        if stride == 2:
            # Branch 1: depthwise 3x3, then pointwise 1x1.
            self.branch1 = nn.Sequential(
                conv_bn(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    groups=in_channels,
                    use_relu=False,
                ),
                conv_bn(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                ),
            )
            branch2_input = in_channels
        else:
            # Half the input channels will pass through unchanged.
            self.branch1 = nn.Identity()
            branch2_input = in_channels // 2

        # Branch 2: pointwise, depthwise, pointwise.
        self.branch2 = nn.Sequential(
            conv_bn(
                branch2_input,
                branch_channels,
                kernel_size=1,
            ),
            conv_bn(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=stride,
                groups=branch_channels,
                use_relu=False,
            ),
            conv_bn(
                branch_channels,
                branch_channels,
                kernel_size=1,
            ),
        )

        self.shuffle = nn.ChannelShuffle(groups=2)

    def forward(self, x):
        if self.stride == 1:
            first_half, second_half = x.chunk(2, dim=1)
            combined = torch.cat(
                (first_half, self.branch2(second_half)),
                dim=1,
            )
        else:
            combined = torch.cat(
                (self.branch1(x), self.branch2(x)),
                dim=1,
            )

        return self.shuffle(combined)


class MalSBSLCNet(nn.Module):
    """MalSBSLCNet for single-channel 64x64 images.

    Structure follows Figure 4 and the supplementary architecture table.
    The number of output classes is configurable for the TRAMIF cohort.

    BatchNorm settings and parameter initialization use PyTorch defaults.
    These are implementation choices unless confirmed by author code.
    """

    def __init__(self, num_classes=51):
        super().__init__()

        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2.")

        self.num_classes = num_classes

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

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten(start_dim=1)
        self.head_norm = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 64, 64):
            raise ValueError("Expected input shape [batch, 1, 64, 64].")

        if not x.is_floating_point():
            raise ValueError("Input must be a floating-point tensor.")

        if self.training and x.shape[0] < 2:
            raise ValueError(
                "Training requires at least two samples because "
                "the classifier uses BatchNorm1d."
            )

        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.head_norm(x)
        x = self.dropout(x)

        # Return logits; do not apply softmax here.
        return self.classifier(x)