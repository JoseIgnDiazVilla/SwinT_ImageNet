from typing import Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

SWIN_CONFIGS = {
    "tiny": {
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": 7,
        "patch_size": 4,
        "mlp_ratio": 4.0,
    },
    "small": {
        "embed_dim": 96,
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": 7,
        "patch_size": 4,
        "mlp_ratio": 4.0,
    },
    "base": {
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "window_size": 7,
        "patch_size": 4,
        "mlp_ratio": 4.0,
    },
}


class SwinForImageClassification(nn.Module):
    """
    Minimal classification wrapper around your SwinTransformer backbone.

    Args:
        backbone: an instance of your SwinTransformer returning multi-scale feature maps
                  (list of feature tensors for stages). This wrapper will pool the
                  last stage (global average) and add a linear head.
        num_classes: number of output classes (ImageNet: 1000)
        dropout: optional dropout before head
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        if hasattr(backbone, "num_features"):
            feat_dim = backbone.num_features
        elif hasattr(backbone, "embed_dim") and hasattr(backbone, "num_layers"):
            feat_dim = int(backbone.embed_dim * 2 ** (backbone.num_layers - 1))
        else:
            feat_dim = 1024

        self.head = nn.Linear(feat_dim, num_classes)

        # initialize head weights
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.backbone(x) 
        if isinstance(features, (list, tuple)):
            last = features[-1]
        else:
            last = features  

        if last.ndim == 5:
            out = last.mean(dim=[2, 3, 4])  # global average pool
        elif last.ndim == 4:
            out = last.mean(dim=[2, 3])
        else:
            raise ValueError(f"Unexpected backbone output dims: {last.shape}")

        out = self.dropout(out)
        logits = self.head(out)
        return logits


def build_swin(variant: str = "tiny", in_chans: int = 3, num_classes: Optional[int] = None, **kwargs) -> nn.Module:

    variant = variant.lower()
    if variant not in SWIN_CONFIGS:
        raise ValueError("Unknown Swin variant: choose from 'tiny','small','base'")

    c = SWIN_CONFIGS[variant]

    swin_kwargs = dict(
        in_chans=in_chans,
        embed_dim=c["embed_dim"],
        window_size=(c["window_size"],) if isinstance(c["window_size"], int) else c["window_size"],
        patch_size=(c["patch_size"],) if isinstance(c["patch_size"], int) else c["patch_size"],
        depths=c["depths"],
        num_heads=c["num_heads"],
        mlp_ratio=c.get("mlp_ratio", 4.0),
    )

    swin_kwargs.update(kwargs)

    backbone = SwinTransformer(
        in_chans=swin_kwargs.pop("in_chans"),
        embed_dim=swin_kwargs.pop("embed_dim"),
        window_size=swin_kwargs.pop("window_size"),
        patch_size=swin_kwargs.pop("patch_size"),
        depths=swin_kwargs.pop("depths"),
        num_heads=swin_kwargs.pop("num_heads"),
        mlp_ratio=swin_kwargs.pop("mlp_ratio"),
        **swin_kwargs,
    )

    if num_classes is not None:
        model = SwinForImageClassification(backbone, num_classes=num_classes, dropout=kwargs.get("dropout", 0.0))
        return model
    return backbone


def swin_tiny(in_chans: int = 3, num_classes: Optional[int] = None, **kwargs):
    return build_swin("tiny", in_chans=in_chans, num_classes=num_classes, **kwargs)


def swin_small(in_chans: int = 3, num_classes: Optional[int] = None, **kwargs):
    return build_swin("small", in_chans=in_chans, num_classes=num_classes, **kwargs)


def swin_base(in_chans: int = 3, num_classes: Optional[int] = None, **kwargs):
    return build_swin("base", in_chans=in_chans, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    try:
        model = swin_tiny(num_classes=1000)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        print("Swin-Tiny forward ok. logits.shape=", logits.shape)
    except Exception as e:
        print("Smoke test failed — make sure SwinTransformer class is in scope. Error:", e)
