# -*- coding: utf-8 -*-
"""BNN 算法文件，包含可选的 XNOR + Popcount 推理层。"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from common import IMAGE_SIZE, create_dataloaders, get_device, set_seed
from torch_pipeline import train_torch_model


EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 0.0005

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "rgbd_bnn_2.pth"
HISTORY_PATH = SCRIPT_DIR / "results" / "rgbd_bnn_history.csv"


class SignSTE(torch.autograd.Function):
    """前向输出 -1/+1；反向只让 [-1,1] 内梯度通过。"""

    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return torch.where(
            input_tensor >= 0,
            torch.ones_like(input_tensor),
            -torch.ones_like(input_tensor),
        )

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        gradient_mask = (input_tensor.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * gradient_mask


def binarize(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


class BinaryLinear(nn.Linear):
    """训练使用：前向传播时临时二值化权重。"""

    def forward(self, input_tensor):
        binary_weight = binarize(self.weight)
        return nn.functional.linear(input_tensor, binary_weight, self.bias)


class XNORPopcountLinear(nn.Linear):
    """推理使用：二值点积 = 2 × Popcount(XNOR) - N。"""

    OUTPUT_CHUNK_SIZE = 64

    def forward(self, input_tensor):
        original_shape = input_tensor.shape
        flat_input = input_tensor.reshape(-1, self.in_features)

        input_bits = flat_input >= 0
        weight_bits = self.weight >= 0
        output_chunks = []

        for start in range(0, self.out_features, self.OUTPUT_CHUNK_SIZE):
            end = min(start + self.OUTPUT_CHUNK_SIZE, self.out_features)
            weight_chunk = weight_bits[start:end]

            xnor_result = torch.eq(
                input_bits.unsqueeze(1),
                weight_chunk.unsqueeze(0),
            )
            popcount = xnor_result.sum(dim=2, dtype=torch.int32)
            binary_dot = (
                2 * popcount - self.in_features
            ).to(input_tensor.dtype)
            output_chunks.append(binary_dot)

        output = torch.cat(output_chunks, dim=1)
        if self.bias is not None:
            output = output + self.bias

        return output.reshape(*original_shape[:-1], self.out_features)


class RGBDBNN(nn.Module):
    """1024 -> 512 -> 256 -> 128 -> classes。"""

    def __init__(self, num_classes: int, use_xnor_popcount: bool = False):
        super().__init__()

        # 第一层输入是连续灰度值，保留 BinaryLinear。
        self.fc1 = BinaryLinear(IMAGE_SIZE * IMAGE_SIZE, 512)
        self.bn1 = nn.BatchNorm1d(512)

        hidden_layer = XNORPopcountLinear if use_xnor_popcount else BinaryLinear

        self.fc2 = hidden_layer(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = hidden_layer(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = hidden_layer(128, num_classes, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = binarize(self.bn1(self.fc1(x)))
        x = binarize(self.bn2(self.fc2(x)))
        x = binarize(self.bn3(self.fc3(x)))
        return self.fc4(x)


def clamp_binary_weights(model: nn.Module) -> None:
    """优化器更新后，把 BinaryLinear 浮点权重限制在 [-1,1]。"""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BinaryLinear):
                module.weight.clamp_(-1.0, 1.0)


def main() -> None:
    set_seed()
    device = get_device()
    print("Using device:", device)

    trainloader, valloader, testloader, class_names = create_dataloaders()

    # 训练时使用普通 BinaryLinear；XNOR 模式用于加载已有参数后推理。
    model = RGBDBNN(
        num_classes=len(class_names),
        use_xnor_popcount=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    train_torch_model(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        class_names=class_names,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_path=MODEL_PATH,
        history_path=HISTORY_PATH,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
        after_optimizer_step=clamp_binary_weights,
    )


if __name__ == "__main__":
    main()
