"""
Model factory. Both backbones are ImageNet-pretrained (transfer learning) with a
fresh 6-output head. Multi-label => 6 independent sigmoids, NOT a 6-way softmax,
because a slice can show several subtypes at once.
"""
import timm

import config as C

# Friendly names -> timm identifiers.
_MODELS = {
    "resnet50":        "resnet50",
    "efficientnet_b0": "efficientnet_b0",   # recommended efficient/deployable choice
    "efficientnet_b3": "efficientnet_b3",
}


def build_model(name: str, pretrained: bool = True):
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(_MODELS)}")
    return timm.create_model(
        _MODELS[name],
        pretrained=pretrained,
        num_classes=C.N_CLASSES,
        in_chans=3,
    )
