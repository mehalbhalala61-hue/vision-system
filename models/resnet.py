# =============================================================================
# models/resnet.py — Full ResNet with SE Blocks
# =============================================================================
# Config-driven: depth, channels, SE, norm, activation — all from YAML.
# num_classes auto-reads from data_config.yaml — zero hardcoding.
# Grad-CAM hook pre-registered on layer4.
#
# Interview note:
#   "I built ResNet from scratch — stem, 4 stages, GlobalAvgPool, FC head.
#    SE blocks add channel attention with minimal overhead. build_model()
#    reads num_classes from data_config.yaml so the same architecture works
#    for any dataset without code changes."
# =============================================================================

import yaml
import logging
import torch
import torch.nn as nn
from typing import Type, Union

from models.blocks import BasicBlock, Bottleneck, get_norm_layer, get_activation

logger = logging.getLogger(__name__)

# Depth → (block_type, [layers_per_stage])
RESNET_CONFIGS: dict[int, tuple] = {
    18:  (BasicBlock,  [2, 2, 2, 2]),
    34:  (BasicBlock,  [3, 4, 6, 3]),
    50:  (Bottleneck,  [3, 4, 6, 3]),
    101: (Bottleneck,  [3, 4, 23, 3]),
}


# =============================================================================
# RESNET
# =============================================================================

class ResNet(nn.Module):
    """
    ResNet with optional Squeeze-Excitation blocks.

    Architecture:
        Stem  : Conv7×7 → BN → ReLU → MaxPool
        Stage1: channels[0]  (no stride — spatial size preserved)
        Stage2: channels[1]  (stride=2 → spatial ÷2)
        Stage3: channels[2]  (stride=2 → spatial ÷2)
        Stage4: channels[3]  (stride=2 → spatial ÷2)
        Head  : GlobalAvgPool → Dropout → Linear(num_classes)

    For input 224×224:
        After stem  : 56×56
        After stage1: 56×56
        After stage2: 28×28
        After stage3: 14×14
        After stage4:  7×7
        After GAP   :  1×1  →  flatten  →  FC

    Args:
        num_classes  : number of output classes
        depth        : 18 | 34 | 50 | 101
        channels     : list of 4 channel widths per stage
        se_block     : enable SE blocks
        reduction    : SE reduction ratio
        norm_layer   : 'batch_norm' | 'group_norm'
        activation   : 'relu' | 'leaky_relu' | 'gelu'
        dropout      : dropout probability in FC head
        zero_init_residual : zero-init last BN in each block
    """

    def __init__(
        self,
        num_classes:        int,
        depth:              int  = 34,
        channels:           list = None,
        se_block:           bool = True,
        reduction:          int  = 16,
        norm_layer:         str  = "batch_norm",
        activation:         str  = "relu",
        dropout:            float = 0.3,
        zero_init_residual: bool  = True,
    ):
        super().__init__()

        if depth not in RESNET_CONFIGS:
            raise ValueError(f"depth={depth} not supported. Choose from {list(RESNET_CONFIGS)}")

        block_type, stage_layers = RESNET_CONFIGS[depth]
        channels = channels or [64, 128, 256, 512]

        self.norm_layer = norm_layer
        self.activation = activation
        self.se_block   = se_block
        self.reduction  = reduction
        self._in_channels = channels[0]   # tracks current channel count across stages

        # ------------------------------------------------------------------
        # STEM
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            get_norm_layer(norm_layer, channels[0]),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ------------------------------------------------------------------
        # 4 STAGES
        # ------------------------------------------------------------------
        self.layer1 = self._make_stage(block_type, channels[0], stage_layers[0], stride=1)
        self.layer2 = self._make_stage(block_type, channels[1], stage_layers[1], stride=2)
        self.layer3 = self._make_stage(block_type, channels[2], stage_layers[2], stride=2)
        self.layer4 = self._make_stage(block_type, channels[3], stage_layers[3], stride=2)

        # ------------------------------------------------------------------
        # HEAD
        # ------------------------------------------------------------------
        final_channels = channels[3] * block_type.expansion
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(final_channels, num_classes),
        )

        # ------------------------------------------------------------------
        # GRAD-CAM HOOK — registered on layer4
        # ------------------------------------------------------------------
        self._gradcam_features: list = []
        self._gradcam_hook      = None
        self._register_gradcam_hook()

        # ------------------------------------------------------------------
        # WEIGHT INIT
        # ------------------------------------------------------------------
        self._init_weights(zero_init_residual)

        logger.info(
            f"ResNet-{depth} | classes={num_classes} | SE={se_block} | "
            f"channels={channels} | params={self.count_params():,}"
        )

    # ------------------------------------------------------------------
    # STAGE BUILDER
    # ------------------------------------------------------------------

    def _make_stage(
        self,
        block_type: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build one stage with num_blocks residual blocks."""
        layers = []
        expanded_out = out_channels * block_type.expansion

        # First block may need downsample projection (stride or channel change)
        downsample = None
        if stride != 1 or self._in_channels != expanded_out:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_channels, expanded_out, kernel_size=1, stride=stride, bias=False),
                get_norm_layer(self.norm_layer, expanded_out),
            )

        # First block (handles stride + channel change)
        layers.append(block_type(
            in_channels  = self._in_channels,
            out_channels = out_channels,
            stride       = stride,
            downsample   = downsample,
            se_block     = self.se_block,
            reduction    = self.reduction,
            norm_layer   = self.norm_layer,
            activation   = self.activation,
        ))
        self._in_channels = expanded_out

        # Remaining blocks (no stride, no channel change)
        for _ in range(1, num_blocks):
            layers.append(block_type(
                in_channels  = self._in_channels,
                out_channels = out_channels,
                stride       = 1,
                downsample   = None,
                se_block     = self.se_block,
                reduction    = self.reduction,
                norm_layer   = self.norm_layer,
                activation   = self.activation,
            ))

        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # WEIGHT INIT
    # ------------------------------------------------------------------

    def _init_weights(self, zero_init_residual: bool) -> None:
        """
        He (Kaiming) init for Conv, standard init for BN.
        Zero-init last BN in residual blocks — 'Bag of Tricks' trick.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and hasattr(m, "bn3"):
                    nn.init.constant_(m.bn3.weight, 0.0)
                elif isinstance(m, BasicBlock) and hasattr(m, "bn2"):
                    nn.init.constant_(m.bn2.weight, 0.0)

    # ------------------------------------------------------------------
    # GRAD-CAM HOOK
    # ------------------------------------------------------------------

    def _register_gradcam_hook(self) -> None:
        """
        Register forward hook on layer4 to capture feature maps.
        Called at __init__ AND after every checkpoint load
        (v3 fix: hooks detach after torch.load — see utils/checkpoint.py).
        """
        # Remove old hook if exists
        if self._gradcam_hook is not None:
            self._gradcam_hook.remove()

        def hook_fn(module, input, output):
            self._gradcam_features.clear()
            self._gradcam_features.append(output)

        self._gradcam_hook = self.layer4.register_forward_hook(hook_fn)
        logger.debug("Grad-CAM hook registered on layer4")

    def get_gradcam_features(self) -> torch.Tensor:
        """Return the last captured layer4 feature maps."""
        if not self._gradcam_features:
            raise RuntimeError(
                "No Grad-CAM features captured yet. Run a forward pass first."
            )
        return self._gradcam_features[0]

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : (B, 3, H, W) — normalised image tensor

        Returns:
            logits : (B, num_classes) — raw scores (no softmax)
        """
        x = self.stem(x)    # (B, 64, 56, 56)
        x = self.layer1(x)  # (B, 64,  56, 56)
        x = self.layer2(x)  # (B, 128, 28, 28)
        x = self.layer3(x)  # (B, 256, 14, 14)
        x = self.layer4(x)  # (B, 512,  7,  7)  ← hook fires here

        x = self.global_avg_pool(x)  # (B, 512, 1, 1)
        logits = self.head(x)        # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def count_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_stage_output_shapes(self, input_size: int = 224) -> dict:
        """
        Returns output spatial size after each stage.
        Useful for architecture diagrams.
        """
        x = torch.zeros(1, 3, input_size, input_size)
        shapes = {}
        with torch.no_grad():
            x = self.stem(x);   shapes["stem"]   = tuple(x.shape)
            x = self.layer1(x); shapes["layer1"] = tuple(x.shape)
            x = self.layer2(x); shapes["layer2"] = tuple(x.shape)
            x = self.layer3(x); shapes["layer3"] = tuple(x.shape)
            x = self.layer4(x); shapes["layer4"] = tuple(x.shape)
            x = self.global_avg_pool(x); shapes["gap"] = tuple(x.shape)
        return shapes


# =============================================================================
# BUILD MODEL — single entry point used by train.py + export.py + API
# =============================================================================

def build_model(
    model_cfg_path: str = "configs/model_config.yaml",
    data_cfg_path:  str = "configs/data_config.yaml",
) -> ResNet:
    """
    Build and return a ResNet model from config files.

    Reads num_classes from data_config.yaml automatically —
    changing the dataset updates the model head with zero code changes.

    Args:
        model_cfg_path : path to model_config.yaml
        data_cfg_path  : path to data_config.yaml

    Returns:
        ResNet model (on CPU — caller moves to device)

    Usage:
        model = build_model()
        model = model.to(device)
    """
    with open(model_cfg_path, encoding='utf-8') as f:
        mcfg = yaml.safe_load(f)
    with open(data_cfg_path, encoding='utf-8') as f:
        dcfg = yaml.safe_load(f)

    num_classes = dcfg["dataset"]["num_classes"]

    model = ResNet(
        num_classes        = num_classes,
        depth              = mcfg["arch"]["depth"],
        channels           = mcfg["arch"]["channels"],
        se_block           = mcfg["blocks"]["se_block"]["enabled"],
        reduction          = mcfg["blocks"]["se_block"]["reduction_ratio"],
        norm_layer         = mcfg["blocks"]["norm_layer"],
        activation         = mcfg["blocks"]["activation"],
        dropout            = mcfg["head"]["dropout"],
        zero_init_residual = mcfg["init"]["zero_init_residual"],
    )

    return model


# =============================================================================
# ENTRYPOINT — quick sanity check
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model = build_model()

    # Forward pass sanity check
    dummy = torch.zeros(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy)

    print(f"\nForward pass output shape: {tuple(logits.shape)}")
    assert logits.shape == (4, model.head[-1].out_features), "Shape mismatch!"

    # Stage shapes
    print("\nOutput shapes per stage:")
    for stage, shape in model.get_stage_output_shapes().items():
        print(f"  {stage:<8}: {shape}")

    print(f"\nTotal params: {model.count_params():,}")
    print("✓ ResNet sanity check passed")
