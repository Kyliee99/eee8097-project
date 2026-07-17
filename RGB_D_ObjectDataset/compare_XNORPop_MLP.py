import os
import time
from pathlib import Path
from rgbd_dataset import RGBDObjectDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. 路径设置
# ============================================================

TEST_DATA_PATH = Path(
    r"D:\personalproject\BNN_Project\data\rgbd-dataset_eval"
)

MLP_MODEL_PATH = Path(
    r"D:\personalproject\BNN_Project\RGB-D_Object_Dataset\models\rgbd_mlp_2.pth"
)

BNN_MODEL_PATH = Path(
    r"D:\personalproject\BNN_Project\RGB-D_Object_Dataset\models\rgbd_bnn_2.pth"
)


# ============================================================
# 2. 基本参数
# 必须与训练时保持一致
# ============================================================

IMAGE_SIZE = 32
BATCH_SIZE = 64

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备:", device)


# ============================================================
# 3. Sign + STE
# BNN模型需要
# ============================================================

class SignSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor):
        return torch.where(
            input_tensor >= 0,
            torch.ones_like(input_tensor),
            -torch.ones_like(input_tensor)
        )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def binarize(x):
    return SignSTE.apply(x)


# ============================================================
# 4. Binary Linear + XNOR/Popcount Linear
# ============================================================

class BinaryLinear(nn.Linear):
    """
    第一层使用。

    原训练模型的第一层输入是连续灰度值，
    只有权重被二值化，因此必须保留原来的计算方式，
    才能直接使用已有.pth训练结果。
    """

    def forward(self, input_tensor):
        binary_weight = binarize(
            self.weight
        )

        return nn.functional.linear(
            input_tensor,
            binary_weight,
            self.bias
        )


class XNORPopcountLinear(nn.Linear):
    """
    第二层及后续层使用。

    这些层的输入已经经过binarize，
    因此输入和权重都是-1/+1，可以使用：

        binary dot product
        = 2 * popcount(XNOR) - input_features

    与原来的二值点积结果数学等价。
    """

    OUTPUT_CHUNK_SIZE = 64

    def forward(self, input_tensor):

        original_shape = input_tensor.shape

        flat_input = input_tensor.reshape(
            -1,
            self.in_features
        )

        # -1 -> False/0
        # +1 -> True/1
        input_bits = flat_input >= 0
        weight_bits = self.weight >= 0

        output_chunks = []

        for start_index in range(
                0,
                self.out_features,
                self.OUTPUT_CHUNK_SIZE
        ):

            end_index = min(
                start_index + self.OUTPUT_CHUNK_SIZE,
                self.out_features
            )

            weight_chunk = weight_bits[
                start_index:end_index
            ]

            xnor_result = torch.eq(
                input_bits.unsqueeze(1),
                weight_chunk.unsqueeze(0)
            )

            popcount_result = xnor_result.sum(
                dim=2,
                dtype=torch.int32
            )

            binary_dot_product = (
                2 * popcount_result
                - self.in_features
            ).to(
                dtype=input_tensor.dtype
            )

            output_chunks.append(
                binary_dot_product
            )

        output_tensor = torch.cat(
            output_chunks,
            dim=1
        )

        if self.bias is not None:
            output_tensor = (
                output_tensor
                + self.bias
            )

        return output_tensor.reshape(
            *original_shape[:-1],
            self.out_features
        )


# ============================================================
# 5. 普通MLP模型
# 必须和训练MLP时的结构完全一致
# ============================================================

class MLP(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(
            IMAGE_SIZE * IMAGE_SIZE,
            512
        )

        self.fc2 = nn.Linear(
            512,
            256
        )

        self.fc3 = nn.Linear(
            256,
            128
        )

        self.fc4 = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        # [batch, 1, 32, 32]
        # 转换为 [batch, 1024]
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = torch.relu(x)

        x = self.fc4(x)

        return x


# ============================================================
# 6. BNN模型
# 必须和训练BNN时的结构完全一致
# ============================================================

class BNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # 第一层输入仍是连续灰度值：
        # 保留原来的二值权重点积，确保兼容已有.pth
        self.fc1 = BinaryLinear(
            IMAGE_SIZE * IMAGE_SIZE,
            512
        )
        self.bn1 = nn.BatchNorm1d(512)

        # 后续层输入已经是-1/+1：
        # 使用XNOR + Popcount替代二值点积
        self.fc2 = XNORPopcountLinear(
            512,
            256
        )
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = XNORPopcountLinear(
            256,
            128
        )
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = XNORPopcountLinear(
            128,
            num_classes,
            bias=False
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = binarize(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = binarize(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = binarize(x)

        x = self.fc4(x)

        return x


# ============================================================
# 7. 测试图片预处理
# 必须与训练阶段保持一致
# ============================================================

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])


# ============================================================
# 8. 加载测试数据集
# ============================================================

if not TEST_DATA_PATH.exists():
    raise FileNotFoundError(
        f"测试数据集路径不存在: {TEST_DATA_PATH}"
    )

test_dataset = RGBDObjectDataset(
    root_dir=str(TEST_DATA_PATH),
    split="test",
    transform=test_transform
)

class_names = test_dataset.class_names
num_classes = len(class_names)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\n类别名称:")
print(class_names)

print("类别数量:", num_classes)
print("测试图片数量:", len(test_dataset))


# ============================================================
# 9. 创建模型
# ============================================================

mlp_model = MLP(
    num_classes=num_classes
).to(device)

bnn_model = BNN(
    num_classes=num_classes
).to(device)


# ============================================================
# 10. 加载checkpoint
# 同时兼容不同保存形式
# ============================================================

def load_model_weights(model, model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # 情况1：
    # torch.save(model.state_dict(), path)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

    # 情况2：
    # torch.save(model, path)
    elif isinstance(checkpoint, nn.Module):
        model.load_state_dict(
            checkpoint.state_dict()
        )

    else:
        raise ValueError(
            f"无法识别模型文件格式: {model_path}"
        )

    model.eval()

    print(f"成功加载模型: {model_path.name}")


load_model_weights(
    mlp_model,
    MLP_MODEL_PATH
)

load_model_weights(
    bnn_model,
    BNN_MODEL_PATH
)


# ============================================================
# 11. 模型测试函数
# ============================================================

def evaluate_model(model, data_loader):

    all_labels = []
    all_predictions = []

    correct = 0
    total = 0

    model.eval()

    # GPU计时前同步
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time

    accuracy = correct / total

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    average_time_per_image = (
        total_time / total
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time": total_time,
        "time_per_image": average_time_per_image,
        "labels": all_labels,
        "predictions": all_predictions
    }


# ============================================================
# 12. 参数数量
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ============================================================
# 13. 统计权重数量
# 不统计bias和BatchNorm参数
# ============================================================

def count_weight_parameters(model):

    total_weights = 0

    for name, parameter in model.named_parameters():

        if "weight" in name and parameter.dim() >= 2:
            total_weights += parameter.numel()

    return total_weights


# ============================================================
# 14. 计算实际文件大小
# ============================================================

def get_file_size_mb(file_path):

    file_size_bytes = os.path.getsize(
        file_path
    )

    return file_size_bytes / (
        1024 * 1024
    )


# ============================================================
# 15. 理论权重大小
# ============================================================

def calculate_theoretical_weight_size(
        number_of_weights,
        bits_per_weight
):

    total_bits = (
        number_of_weights *
        bits_per_weight
    )

    total_bytes = total_bits / 8

    total_mb = total_bytes / (
        1024 * 1024
    )

    return total_mb


# ============================================================
# 16. 开始比较
# ============================================================

print("\n正在测试MLP...")

mlp_results = evaluate_model(
    mlp_model,
    test_loader
)

print("正在测试BNN...")

bnn_results = evaluate_model(
    bnn_model,
    test_loader
)


# ============================================================
# 17. 参数和模型大小
# ============================================================

mlp_total_params = count_parameters(
    mlp_model
)

bnn_total_params = count_parameters(
    bnn_model
)

mlp_weight_count = count_weight_parameters(
    mlp_model
)

bnn_weight_count = count_weight_parameters(
    bnn_model
)

mlp_file_size = get_file_size_mb(
    MLP_MODEL_PATH
)

bnn_file_size = get_file_size_mb(
    BNN_MODEL_PATH
)

# MLP权重按照FP32，即32 bit
mlp_theoretical_size = (
    calculate_theoretical_weight_size(
        mlp_weight_count,
        bits_per_weight=32
    )
)

# BNN权重理论按照1 bit
bnn_theoretical_size = (
    calculate_theoretical_weight_size(
        bnn_weight_count,
        bits_per_weight=1
    )
)


# ============================================================
# 18. 输出比较结果
# ============================================================

print("\n")
print("=" * 75)
print("MLP 与 BNN 模型比较结果")
print("=" * 75)

print(
    f"{'指标':<25}"
    f"{'MLP':>20}"
    f"{'BNN':>20}"
)

print("-" * 75)

print(
    f"{'Accuracy':<25}"
    f"{mlp_results['accuracy'] * 100:>19.2f}%"
    f"{bnn_results['accuracy'] * 100:>19.2f}%"
)

print(
    f"{'Precision':<25}"
    f"{mlp_results['precision']:>20.4f}"
    f"{bnn_results['precision']:>20.4f}"
)

print(
    f"{'Recall':<25}"
    f"{mlp_results['recall']:>20.4f}"
    f"{bnn_results['recall']:>20.4f}"
)

print(
    f"{'F1-score':<25}"
    f"{mlp_results['f1']:>20.4f}"
    f"{bnn_results['f1']:>20.4f}"
)

print(
    f"{'参数数量':<25}"
    f"{mlp_total_params:>20,}"
    f"{bnn_total_params:>20,}"
)

print(
    f"{'权重数量':<25}"
    f"{mlp_weight_count:>20,}"
    f"{bnn_weight_count:>20,}"
)

print(
    f"{'模型文件大小 MB':<25}"
    f"{mlp_file_size:>20.4f}"
    f"{bnn_file_size:>20.4f}"
)

print(
    f"{'理论权重大小 MB':<25}"
    f"{mlp_theoretical_size:>20.4f}"
    f"{bnn_theoretical_size:>20.4f}"
)

print(
    f"{'总推理时间 秒':<25}"
    f"{mlp_results['total_time']:>20.6f}"
    f"{bnn_results['total_time']:>20.6f}"
)

print(
    f"{'单张推理时间 毫秒':<25}"
    f"{mlp_results['time_per_image'] * 1000:>20.6f}"
    f"{bnn_results['time_per_image'] * 1000:>20.6f}"
)

print("=" * 75)


# ============================================================
# 19. 输出每个类别的详细结果
# ============================================================

print("\nMLP详细分类报告:")

print(
    classification_report(
        mlp_results["labels"],
        mlp_results["predictions"],
        target_names=class_names,
        zero_division=0
    )
)

print("\nBNN详细分类报告:")

print(
    classification_report(
        bnn_results["labels"],
        bnn_results["predictions"],
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 20. 输出混淆矩阵
# ============================================================

mlp_confusion_matrix = confusion_matrix(
    mlp_results["labels"],
    mlp_results["predictions"]
)

bnn_confusion_matrix = confusion_matrix(
    bnn_results["labels"],
    bnn_results["predictions"]
)

print("\nMLP混淆矩阵:")
print(mlp_confusion_matrix)

print("\nBNN混淆矩阵:")
print(bnn_confusion_matrix)


# ============================================================
# 21. 自动计算差异
# ============================================================

accuracy_difference = (
    mlp_results["accuracy"] -
    bnn_results["accuracy"]
)

theoretical_compression_ratio = (
    mlp_theoretical_size /
    bnn_theoretical_size
)

print("\n最终分析:")

print(
    f"MLP与BNN准确率差值: "
    f"{accuracy_difference * 100:.2f}%"
)

print(
    f"BNN理论权重压缩比例: "
    f"{theoretical_compression_ratio:.2f}倍"
)

if mlp_results["accuracy"] > bnn_results["accuracy"]:
    print(
        "MLP分类准确率更高。"
    )

elif mlp_results["accuracy"] < bnn_results["accuracy"]:
    print(
        "BNN分类准确率更高。"
    )

else:
    print(
        "MLP和BNN分类准确率相同。"
    )

print(
    "\n注意：PyTorch中的BNN仍可能使用浮点矩阵运算，"
    "因此软件推理速度不一定比MLP快。"
)

print(
    "BNN真正的计算优势主要体现在使用"
    "XNOR和Popcount的FPGA或ASIC硬件实现中。"
)