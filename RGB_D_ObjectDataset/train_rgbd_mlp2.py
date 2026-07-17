import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.font_manager import weight_dict
from torch.utils.data import DataLoader
from torchvision import transforms

from rgbd_dataset import RGBDObjectDataset


# ============================================================
# 1. 固定随机种子
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ============================================================
# 2. 设置设备
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. 图像预处理
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])


# ============================================================
# 4. 创建训练集、验证集和测试集
# ============================================================

dataset_root = "../data/rgbd-dataset_eval"

train_dataset = RGBDObjectDataset(
    root_dir=dataset_root,
    split="train",
    transform=transform
)

val_dataset = RGBDObjectDataset(
    root_dir=dataset_root,
    split="val",
    transform=transform
)

test_dataset = RGBDObjectDataset(
    root_dir=dataset_root,
    split="test",
    transform=transform
)


print("Number of classes:", len(train_dataset.class_names))
print("Classes:", train_dataset.class_names)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# 5. 创建DataLoader
# ============================================================

batch_size = 32

trainloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

valloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

testloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# 6. 定义普通MLP
# 输入：32×32 = 1024
# 输出：13个类别
# ============================================================

class RGBDMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 13)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)                      #训练时随机关闭30%的神经元

    def forward(self, x):

        # [batch, 1, 32, 32]
        # 变为 [batch, 1024]
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        # 最后一层不使用ReLU
        x = self.fc4(x)

        return x


model = RGBDMLP().to(device)

print(model)


# ============================================================
# 7. 损失函数和优化器
# ============================================================

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4                      # 就是0.0001，它会在训练时对过大的群众进行惩罚
)


# ============================================================
# 8. 评估函数
# ============================================================

def evaluate(model, dataloader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    average_loss = (
        running_loss / len(dataloader)
    )

    accuracy = (
        100 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# 9. 训练模型
# ============================================================

epochs = 30
best_val_accuracy = 0.0

patience = 5                             # 如果验证集表现连续 5 个 Epoch 没有提升，就停止训练。
epochs_without_improvement = 0           #

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

model_path = (
    model_dir / "rgbd_mlp_2.pth"
)


for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        # 清空上一轮梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(
            outputs,
            labels
        )

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = (
        running_loss / len(trainloader)
    )

    train_accuracy = (
        100 * correct / total
    )

    val_loss, val_accuracy = evaluate(
        model,
        valloader
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_accuracy:.2f}%, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    # 保存验证集准确率最高的模型
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0          # 当前验证准确率提高，连续无提升次数清零

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"Best model saved: "
            f"{best_val_accuracy:.2f}%"
        )

    else:

        epochs_without_improvement += 1          # 只有没有提高时才加1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)"
        )

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

# ============================================================
# 10. 加载最佳模型
# ============================================================

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("\nBest MLP model loaded.")


# ============================================================
# 11. 测试最佳模型
# ============================================================

test_loss, test_accuracy = evaluate(
    model,
    testloader
)

print("\nFinal MLP Result")
print("------------------------------")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Model saved to: {model_path.resolve()}")