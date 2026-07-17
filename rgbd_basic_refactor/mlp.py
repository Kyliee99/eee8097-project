# -*- coding: utf-8 -*-
"""MLP 算法文件。"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from common import IMAGE_SIZE, create_dataloaders, get_device, set_seed
from torch_pipeline import train_torch_model


EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "rgbd_mlp_2.pth"
HISTORY_PATH = SCRIPT_DIR / "results" / "rgbd_mlp_history.csv"


class RGBDMLP(nn.Module):
    """1024 -> 512 -> 256 -> 128 -> classes。"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


def main() -> None:
    set_seed()
    device = get_device()
    print("Using device:", device)

    trainloader, valloader, testloader, class_names = create_dataloaders()

    model = RGBDMLP(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_torch_model(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        class_names=class_names,
        criterion=criterion,
        optimizer=optimizer,
        model_path=MODEL_PATH,
        history_path=HISTORY_PATH,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
    )


if __name__ == "__main__":
    main()
