# =============================================================================
# models/blocks.py — ResNet Building Blocks
# =============================================================================
# Contains:
#   SEBlock      — Squeeze-Excitation channel attention
#   BasicBlock   — 2-conv residual block  (ResNet-18 / 34)
#   Bottleneck   — 3-conv residual block  (ResNet-50 / 101)
#
# Interview notes:
#   SE Block  : "Channel attention — model learns WHICH feature maps matter.
#                Only +2% params but +1-2% accuracy. Used in SENet (ImageNet
#                winner 2017) and EfficientNet."
#   BasicBlock: "Two 3×3 convs with a skip connection. BN before ReLU
#                (pre-activation style improves gradient flow)."
#   Bottleneck : "1×1 → 3×3 → 1×1 with 4× channel expansion. Efficient for
#                deeper networks — same FLOPs as BasicBlock at larger depth."
#   Zero-init  : "Zero-initializing the last BN in each block so residual
#                 branches start as identity — proved to improve accuracy
#                 (He et al., 2019 'Bag of Tricks')."
# =============================================================================

import torch
import torch.nn as nn
from typing import Optional, Callable


# =============================================================================
# HELPERS
# =============================================================================

def get_norm_layer(norm_type: str, num_features: int) -> nn.Module:
    """
    Return the requested normalisation layer.
    norm_type: 'batch_norm' | 'group_norm' | 'layer_norm'
    """
    if norm_type == "batch_norm":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "group_norm":
        num_groups = min(32, num_features // 4)
        return nn.GroupNorm(num_groups, num_features)
    elif norm_type == "layer_norm":
        return nn.LayerNorm(num_features)
    else:
        raise ValueError(f"Unknown norm_layer: {norm_type!r}. Use batch_norm | group_norm | layer_norm")


def get_activation(act_type: str) -> nn.Module:
    """Return activation module by name."""
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {act_type!r}. Use relu | leaky_relu | gelu")


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    """3×3 conv with padding — no bias (BN handles bias)."""
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1,
        groups=groups, bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1×1 conv — used for channel projection / Bottleneck."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


# =============================================================================
# SQUEEZE-EXCITATION BLOCK
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-Excitation channel attention (Hu et al., 2018).

    Pipeline:
        x  →  GlobalAvgPool  →  FC(C/r)  →  ReLU  →  FC(C)  →  Sigmoid
           →  channel-wise multiply with x

    Args:
        channels        : number of input channels C
        reduction_ratio : bottleneck ratio r (default 16)

    Why it works:
        The network learns to re-weight channels — suppressing noise channels
        and amplifying informative ones. Adds only 2/r × C² parameters.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction_ratio, 4)  # floor at 4

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),                        # (B, C, 1, 1) → (B, C)
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),                        # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # se_weights: (B, C) → unsqueeze → (B, C, 1, 1) for broadcast
        se_weights = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * se_weights                   # channel-wise re-scaling


# =============================================================================
# BASIC BLOCK — ResNet-18 / 34
# =============================================================================

class BasicBlock(nn.Module):
    """
    Standard 2-conv residual block for ResNet-18 and ResNet-34.

    Structure:
        conv3x3 → BN → ReLU → conv3x3 → BN → [SE] → + skip → ReLU

    Args:
        in_channels   : input channels
        out_channels  : output channels
        stride        : stride for first conv (2 = downsample)
        downsample    : optional projection shortcut for channel mismatch
        se_block      : whether to attach SE block after second BN
        reduction     : SE reduction ratio
        norm_layer    : norm constructor (default BatchNorm2d)
        activation    : activation type string
    """

    expansion: int = 1   # output channels = out_channels × expansion

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int = 1,
        downsample:   Optional[nn.Module] = None,
        se_block:     bool = True,
        reduction:    int = 16,
        norm_layer:   str = "batch_norm",
        activation:   str = "relu",
    ):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1   = get_norm_layer(norm_layer, out_channels)
        self.act   = get_activation(activation)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2   = get_norm_layer(norm_layer, out_channels)

        # SE block after second BN (before residual addition)
        self.se = SEBlock(out_channels, reduction) if se_block else nn.Identity()

        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x                         # save for skip connection

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                   # channel attention (or identity)

        if self.downsample is not None:
            identity = self.downsample(x)    # project skip if shape differs

        out = self.act(out + identity)       # residual addition + activation
        return out


# =============================================================================
# BOTTLENECK BLOCK — ResNet-50 / 101
# =============================================================================

class Bottleneck(nn.Module):
    """
    3-conv bottleneck residual block for ResNet-50+.

    Structure:
        conv1×1 → BN → ReLU
        conv3×3 → BN → ReLU
        conv1×1 → BN → [SE] → + skip → ReLU

    The 1×1 convs compress/expand channels, so the expensive 3×3 conv
    operates on C/4 channels — same FLOPs, 4× more representational depth.

    Args: same as BasicBlock, expansion is always 4.
    """

    expansion: int = 4   # output channels = out_channels × 4

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int = 1,
        downsample:   Optional[nn.Module] = None,
        se_block:     bool = True,
        reduction:    int = 16,
        norm_layer:   str = "batch_norm",
        activation:   str = "relu",
    ):
        super().__init__()
        expanded = out_channels * self.expansion

        # 1×1 compress
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1   = get_norm_layer(norm_layer, out_channels)

        # 3×3 spatial
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2   = get_norm_layer(norm_layer, out_channels)

        # 1×1 expand
        self.conv3 = conv1x1(out_channels, expanded)
        self.bn3   = get_norm_layer(norm_layer, expanded)

        self.act = get_activation(activation)
        self.se  = SEBlock(expanded, reduction) if se_block else nn.Identity()

        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.act(out + identity)
        return out
