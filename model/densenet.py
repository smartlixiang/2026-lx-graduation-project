"""DenseNet models for image classification."""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, inter_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inter_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = float(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0.0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], dim=1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[tuple[str, nn.Module]] = []
        num_features = num_input_features
        for idx in range(num_layers):
            layer = _DenseLayer(
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            layers.append((f"denselayer{idx + 1}", layer))
            num_features += growth_rate
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _Transition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.norm(x)))
        return self.pool(x)


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        compression: float = 0.5,
        drop_rate: float = 0.0,
        num_classes: int = 10,
        small_inputs: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < compression <= 1.0:
            raise ValueError("compression must be in (0, 1].")

        if small_inputs:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv2d(
                                3,
                                num_init_features,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                        ),
                        ("norm0", nn.BatchNorm2d(num_init_features)),
                        ("relu0", nn.ReLU(inplace=True)),
                    ]
                )
            )
        else:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv2d(
                                3,
                                num_init_features,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False,
                            ),
                        ),
                        ("norm0", nn.BatchNorm2d(num_init_features)),
                        ("relu0", nn.ReLU(inplace=True)),
                        ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    ]
                )
            )

        num_features = num_init_features
        for idx, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module(f"denseblock{idx + 1}", block)
            num_features += num_layers * growth_rate

            if idx != len(block_config) - 1:
                next_features = int(num_features * compression)
                transition = _Transition(num_input_features=num_features, num_output_features=next_features)
                self.features.add_module(f"transition{idx + 1}", transition)
                num_features = next_features

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def densenet121(num_classes: int = 10) -> DenseNet:
    return DenseNet(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        compression=0.5,
        drop_rate=0.0,
        num_classes=num_classes,
        small_inputs=True,
    )
