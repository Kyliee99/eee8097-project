import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from rgbd_dataset import RGBDObjectDataset


# ============================================================
# 1. 固定随机种子
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    固定随机种子，使每次运行结果尽量保持一致。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 尽量保证GPU运算可重复
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ============================================================
# 2. 设置CPU或GPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. 设置文件路径
# ============================================================

# 当前代码文件所在目录：
# .../BNN_Project/RGB-D Object Dataset
script_dir = Path(__file__).resolve().parent

# 项目根目录：
# .../BNN_Project
project_root = script_dir.parent

# 数据集目录：
# .../BNN_Project/data/rgbd-dataset_eval
dataset_root = (
    project_root
    / "data"
    / "rgbd-dataset_eval"
)

# 模型保存目录：
# .../BNN_Project/RGB-D Object Dataset/models
model_dir = script_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

# 训练结果保存目录
results_dir = script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# 最佳BNN模型保存位置
model_path = model_dir / "rgbd_bnn_2.pth"

# 训练历史CSV保存位置
history_path = results_dir / "rgbd_bnn_history.csv"


print("Dataset path:", dataset_root.resolve())
print("Dataset exists:", dataset_root.exists())

if not dataset_root.exists():
    raise FileNotFoundError(
        f"找不到数据集目录：{dataset_root.resolve()}"
    )


# ============================================================
# 4. 图像预处理
# ============================================================

# 为了和当前MLP公平比较： RGB图片 → 灰度图 → 32×32 → Tensor → [-1, 1]
transform = transforms.Compose([                          #原始RGB图片
    transforms.Grayscale(num_output_channels=1),          #转成1通道灰度图
    transforms.Resize((32, 32)),                          #统一缩放为32×32
    transforms.ToTensor(),                                #转为Tensor
    transforms.Normalize(                                 #像素从[0,1]映射到[-1,1]
        mean=(0.5,),
        std=(0.5,)
    )
])


# ============================================================
# 5. 创建训练集、验证集和测试集
# ============================================================

train_dataset = RGBDObjectDataset(
    root_dir=str(dataset_root),
    split="train",
    transform=transform
)

val_dataset = RGBDObjectDataset(
    root_dir=str(dataset_root),
    split="val",
    transform=transform
)

test_dataset = RGBDObjectDataset(
    root_dir=str(dataset_root),
    split="test",
    transform=transform
)


# 自动获取类别数量
number_of_classes = len(
    train_dataset.class_names
)

print("\nDataset information")
print("------------------------------")
print("Number of classes:", number_of_classes)
print("Class names:", train_dataset.class_names)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# 6. 创建DataLoader
# ============================================================

batch_size = 32

trainloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

valloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

testloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 7. 定义Sign + STE
# ============================================================

class SignSTE(torch.autograd.Function):
    """
    Clipped Straight-Through Estimator。

    前向传播：
        输入 >= 0 -> +1
        输入 < 0  -> -1

    反向传播：
        只有输入位于 [-1, 1] 时允许梯度通过。
    """

    @staticmethod
    def forward(ctx, input_tensor):

        # 保存前向输入，供backward使用
        ctx.save_for_backward(input_tensor)

        return torch.where(
            input_tensor >= 0,
            torch.ones_like(input_tensor),
            -torch.ones_like(input_tensor)
        )

    @staticmethod
    def backward(ctx, grad_output):

        # 取回forward保存的输入
        input_tensor, = ctx.saved_tensors

        # |input| <= 1时为True，否则为False
        gradient_mask = (
            input_tensor.abs() <= 1.0
        ).to(grad_output.dtype)

        # 只允许范围[-1,1]内的梯度通过
        return grad_output * gradient_mask

def binarize(x):
    """
    对Tensor进行二值化。
    """
    return SignSTE.apply(x)


# ============================================================
# 8. 定义BinaryLinear
# ============================================================

class BinaryLinear(nn.Linear):
    """
    二值全连接层。

    .pth文件中保留浮点权重用于训练；
    前向传播时将权重临时转换成-1或+1。
    """

    def forward(self, input_tensor):

        binary_weight = binarize(
            self.weight
        )

        return nn.functional.linear(
            input_tensor, binary_weight, self.bias
        )


# ============================================================
# 9. 定义RGB-D BNN模型
# ============================================================

class RGBDBNN(nn.Module):
    """
    网络结构：

    32×32灰度图
        ↓
    1024
        ↓
    512
        ↓
    256
        ↓
    128
        ↓
    number_of_classes
    """

    def __init__(self, num_classes):
        super().__init__()

        # 第一层：1024 -> 512
        self.fc1 = BinaryLinear(
            32 * 32,
            512
        )
        self.bn1 = nn.BatchNorm1d(512)

        # 第二层：512 -> 256
        self.fc2 = BinaryLinear(
            512,
            256
        )
        self.bn2 = nn.BatchNorm1d(256)

        # 第三层：256 -> 128
        self.fc3 = BinaryLinear(
            256,
            128
        )
        self.bn3 = nn.BatchNorm1d(128)

        # 输出层：128 -> 类别数量
        self.fc4 = BinaryLinear(
            128,
            num_classes,
            bias=False
        )

    def forward(self, x):
        # [batch, 1, 32, 32]
        # -> [batch, 1024]
        x = x.view(x.size(0), -1)

        # BinaryLinear -> BatchNorm -> Sign
        x = self.fc1(x)
        x = self.bn1(x)
        x = binarize(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = binarize(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = binarize(x)

        # 最后一层不做Sign
        x = self.fc4(x)

        return x


# 创建模型
model = RGBDBNN(
    num_classes=number_of_classes
).to(device)

print("\nBNN model")
print("------------------------------")
print(model)


# ============================================================
# 10. 损失函数与优化器
# ============================================================

criterion = nn.CrossEntropyLoss()

learning_rate = 0.0005

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,

    # 验证准确率越大越好
    mode="max",

    # 学习率乘以0.5
    factor=0.5,

    # 连续2轮没有改善，就降低学习率
    patience=2,

    # 学习率最低值
    min_lr=1e-6
)

# ============================================================
# 11. 定义评估函数
# ============================================================

def evaluate(
    current_model,
    dataloader
):
    """
    在验证集或测试集上计算：

    1. 平均Loss
    2. Accuracy
    """

    current_model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # 前向传播
            outputs = current_model(images)

            # 计算Loss
            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            # 找到最大分数对应的类别
            predicted = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    if len(dataloader) == 0:
        raise RuntimeError(
            "DataLoader中没有数据。"
        )

    average_loss = (
        running_loss
        / len(dataloader)
    )

    accuracy = (
        100.0
        * correct
        / total
    )

    return average_loss, accuracy


# ============================================================
# 12. 设置训练参数
# ============================================================

epochs = 50                              # 30-->50

# 历史最佳验证准确率
best_val_accuracy = 0.0

# Early Stopping：
# 连续5轮没有提升就停止
patience = 8                              # 5-->8

# 当前连续没有提升的Epoch数量
epochs_without_improvement = 0

# 保存每一轮结果
training_history = []


# ============================================================
# 13. 开始训练BNN
# ============================================================

print("\nStart BNN training")
print("==============================")


for epoch in range(epochs):

    # 切换到训练模式
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # 清空上一轮梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算Loss
        loss = criterion(
            outputs,
            labels
        )

        # 反向传播
        loss.backward()

        # 更新浮点权重
        optimizer.step()

        # 将BinaryLinear中的浮点权重限制在[-1, 1]
        with torch.no_grad():                          ## 权重裁剪: 只裁剪BinaryLinear中的weight, 不裁剪bias和BatchNorm参数

            for module in model.modules():

                if isinstance(module, BinaryLinear):
                    module.weight.clamp_(
                        -1.0,
                        1.0
                    )

        # 累计Loss
        running_loss += loss.item()

        # 当前Batch预测结果
        predicted = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    # 当前Epoch训练Loss
    train_loss = (
        running_loss
        / len(trainloader)
    )

    # 当前Epoch训练准确率
    train_accuracy = (
        100.0
        * correct
        / total
    )

    # 验证集结果
    val_loss, val_accuracy = evaluate(
        model,
        valloader
    )

    # 根据验证准确率自动调整学习率
    scheduler.step(val_accuracy)

    # 查看当前学习率
    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    # 记录历史结果
    training_history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_learning_rate
    })

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_accuracy:.2f}%, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val Accuracy: {val_accuracy:.2f}%"
        f"LR: {current_learning_rate:.6f}"
    )

    # ========================================================
    # 保存最佳模型 + Early Stopping
    # ========================================================

    if val_accuracy > best_val_accuracy:

        # 更新历史最佳验证准确率
        best_val_accuracy = val_accuracy

        # 当前轮有提高，计数器清零
        epochs_without_improvement = 0

        # 保存最佳模型参数
        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"Best BNN model saved: "
            f"{best_val_accuracy:.2f}%"
        )

    else:

        # 当前验证准确率没有提高
        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)"
        )

        # 连续5轮没有提高，提前停止
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break


# ============================================================
# 14. 保存训练历史到CSV
# ============================================================

with open(
    history_path,
    mode="w",
    newline="",
    encoding="utf-8"
) as csv_file:

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(training_history)


print("\nTraining history saved to:")
print(history_path.resolve())


# ============================================================
# 15. 加载验证集表现最好的BNN模型
# ============================================================

if not model_path.exists():
    raise FileNotFoundError(
        f"没有找到最佳BNN模型：{model_path.resolve()}"
    )

best_state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    best_state_dict
)

model.eval()

print("\nBest BNN model loaded.")


# ============================================================
# 16. 在测试集上评估最佳模型
# ============================================================

test_loss, test_accuracy = evaluate(
    model,
    testloader
)


# ============================================================
# 17. 输出最终结果
# ============================================================

print("\nFinal BNN Result")
print("------------------------------")
print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)
print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)
print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)
print(
    f"Model saved to: "
    f"{model_path.resolve()}"
)


# ============================================================
# 18. 输出参数数量
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nModel parameters")
print("------------------------------")
print("Total parameters:", total_parameters)
print("Trainable parameters:", trainable_parameters)


# ============================================================
# 19. 检查最后一层的权重
# ============================================================

with torch.no_grad():

    binary_fc4_weight = binarize(
        model.fc4.weight
    )

print("\nFinal layer information")
print("------------------------------")
print(
    "fc4 weight shape:",
    model.fc4.weight.shape
)
print(
    "Binary fc4 unique values:",
    torch.unique(binary_fc4_weight)
)