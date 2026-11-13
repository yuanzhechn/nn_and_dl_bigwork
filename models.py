"""Independent LeNet-5 implementation and controlled architectural variants."""

import json

import torch
from torch import nn


VARIANTS = ("baseline", "dw", "se", "gelu", "dw_se")
INPUT_SHAPE = (1, 28, 28)


class SqueezeExcitation(nn.Module):
    """Learn one sigmoid gate per channel from spatial channel averages."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = channels // reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(), nn.Linear(channels, hidden), nn.ReLU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.excitation(self.squeeze(x))
        return x * weights[:, :, None, None]


class LeNet5(nn.Module):
    """LeNet-5 geometry with dense C3 connectivity and a logits classifier."""

    def __init__(self, *, depthwise: bool, attention: bool, gelu: bool):
        super().__init__()
        activation = nn.GELU if gelu else nn.ReLU
        if depthwise:
            second_conv = nn.Sequential(
                nn.Conv2d(6, 6, 5, groups=6, bias=False),
                nn.Conv2d(6, 16, 1),
            )
        else:
            second_conv = nn.Conv2d(6, 16, 5)
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), activation(), nn.AvgPool2d(2),
            second_conv, activation(),
            SqueezeExcitation(16) if attention else nn.Identity(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(400, 120), activation(),
            nn.Linear(120, 84), activation(), nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(variant: str) -> LeNet5:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model variant: {variant!r}")
    return LeNet5(
        depthwise=variant in {"dw", "dw_se"},
        attention=variant in {"se", "dw_se"},
        gelu=variant == "gelu",
    )


@torch.inference_mode()
def model_cost(model: nn.Module) -> dict:
    """Count trainable parameters and Conv2d/Linear MACs for one image."""
    layers = []

    def count_macs(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            fan_in = module.in_channels // module.groups
            fan_in *= module.kernel_size[0] * module.kernel_size[1]
        else:
            fan_in = module.in_features
        layers.append({
            "type": type(module).__name__,
            "output_shape": list(output.shape[1:]),
            "macs": output.numel() * fan_in,
        })

    hooks = [
        module.register_forward_hook(count_macs) for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    try:
        parameter = next(model.parameters())
        model(parameter.new_zeros((1, *INPUT_SHAPE)))
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "macs": sum(layer["macs"] for layer in layers),
        "layers": layers,
    }


if __name__ == "__main__":
    print(json.dumps({variant: model_cost(build_model(variant)) for variant in VARIANTS}, indent=2))
