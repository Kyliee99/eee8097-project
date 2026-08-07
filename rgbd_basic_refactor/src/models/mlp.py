"""Full-precision MLP baseline."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only in lightweight environments
    torch = None
    nn = None


if nn is not None:
    class RGBDMLP(nn.Module):
        def __init__(self, image_size: int, num_classes: int, hidden_sizes=(512, 256, 128)):
            super().__init__()
            if len(hidden_sizes) != 3:
                raise ValueError("This checkpoint-compatible MLP requires exactly three hidden sizes")
            first, second, third = map(int, hidden_sizes)
            # Attribute names intentionally match the existing rgbd_mlp_2.pth state_dict.
            self.fc1 = nn.Linear(image_size * image_size, first)
            self.fc2 = nn.Linear(first, second)
            self.fc3 = nn.Linear(second, third)
            self.fc4 = nn.Linear(third, num_classes)

        def forward(self, inputs):
            values = inputs.reshape(inputs.shape[0], -1)
            values = torch.relu(self.fc1(values))
            values = torch.relu(self.fc2(values))
            values = torch.relu(self.fc3(values))
            return self.fc4(values)
else:
    class RGBDMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to construct RGBDMLP")
