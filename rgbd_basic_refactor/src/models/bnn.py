"""BNN layers and mathematically equivalent XNOR-Popcount inference."""

from __future__ import annotations

import numpy as np


def binary_dot_from_bits(input_bits: np.ndarray, weight_bits: np.ndarray) -> np.ndarray:
    """Reference identity: dot({-1,+1}) = 2*popcount(XNOR)-N."""
    inputs = np.asarray(input_bits, dtype=bool)
    weights = np.asarray(weight_bits, dtype=bool)
    if inputs.shape[-1] != weights.shape[-1]:
        raise ValueError("Input and weight feature dimensions differ")
    matches = np.equal(inputs[..., None, :], weights)
    return 2 * matches.sum(axis=-1, dtype=np.int64) - inputs.shape[-1]


try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if torch is not None:
    class SignSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            ctx.save_for_backward(inputs)
            return torch.where(inputs >= 0, torch.ones_like(inputs), -torch.ones_like(inputs))

        @staticmethod
        def backward(ctx, gradient):
            (inputs,) = ctx.saved_tensors
            return gradient * (inputs.abs() <= 1).to(gradient.dtype)


    def binarize(values):
        return SignSTE.apply(values)


    class BinaryLinear(nn.Linear):
        def forward(self, inputs):
            return nn.functional.linear(inputs, binarize(self.weight), self.bias)


    class XNORPopcountLinear(nn.Linear):
        """Boolean-tensor reference implementation; it is not packed-bit hardware."""

        def __init__(self, *args, output_chunk_size: int = 64, **kwargs):
            super().__init__(*args, **kwargs)
            self.output_chunk_size = output_chunk_size

        def forward(self, inputs):
            original_shape = inputs.shape
            flat = inputs.reshape(-1, self.in_features)
            input_bits, weight_bits = flat >= 0, self.weight >= 0
            chunks = []
            for start in range(0, self.out_features, self.output_chunk_size):
                weights = weight_bits[start : start + self.output_chunk_size]
                matches = torch.eq(input_bits.unsqueeze(1), weights.unsqueeze(0))
                count = matches.sum(dim=2, dtype=torch.int32)
                chunks.append((2 * count - self.in_features).to(inputs.dtype))
            output = torch.cat(chunks, dim=1)
            if self.bias is not None:
                output = output + self.bias
            return output.reshape(*original_shape[:-1], self.out_features)


    class RGBDBNN(nn.Module):
        def __init__(self, image_size: int, num_classes: int, hidden_sizes=(512, 256, 128), use_xnor=False):
            super().__init__()
            if len(hidden_sizes) != 3:
                raise ValueError("This checkpoint-compatible BNN requires exactly three hidden sizes")
            first, second, third = map(int, hidden_sizes)
            hidden_layer = XNORPopcountLinear if use_xnor else BinaryLinear
            # Attribute names intentionally match the existing rgbd_bnn_2.pth state_dict.
            self.fc1 = BinaryLinear(image_size * image_size, first)
            self.bn1 = nn.BatchNorm1d(first)
            self.fc2 = hidden_layer(first, second)
            self.bn2 = nn.BatchNorm1d(second)
            self.fc3 = hidden_layer(second, third)
            self.bn3 = nn.BatchNorm1d(third)
            self.fc4 = hidden_layer(third, num_classes, bias=False)

        def forward(self, inputs):
            values = inputs.reshape(inputs.shape[0], -1)
            values = binarize(self.bn1(self.fc1(values)))
            values = binarize(self.bn2(self.fc2(values)))
            values = binarize(self.bn3(self.fc3(values)))
            return self.fc4(values)


    def clamp_binary_weights(model) -> None:
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, BinaryLinear):
                    module.weight.clamp_(-1.0, 1.0)
else:
    def binarize(values):  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for BNN inference")

    class BinaryLinear:  # type: ignore[no-redef]
        pass

    class XNORPopcountLinear:  # type: ignore[no-redef]
        pass

    class RGBDBNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to construct RGBDBNN")

    def clamp_binary_weights(model) -> None:  # type: ignore[no-redef]
        raise RuntimeError("PyTorch is required for BNN training")
