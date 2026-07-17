import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms


# ============================================================
# 1. 项目路径
# ============================================================

project_root = Path(
    r"D:\personalproject\BNN_Project"
)

dataset_folder = (
    project_root
    / "data"
    / "rgbd-dataset_eval"
    / "apple"
    / "apple_1"
)


# ============================================================
# 2. 自动查找已经训练好的模型
# ============================================================

model_files = list(
    project_root.rglob("rgbd_bnn_2.pth")
)

if not model_files:
    raise FileNotFoundError(
        f"在项目中没有找到 rgbd_bnn_2.pth：{project_root}"
    )

model_path = model_files[0]

print("找到模型文件：")
print(model_path)


# ============================================================
# 3. 直接读取state_dict
#
# 不需要重新定义BNN模型。
# state_dict中保存了每一层的参数Tensor。
# ============================================================

state_dict = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True
)

print("\n模型权重读取成功")


# ============================================================
# 4. 检查第一层权重名称
#
# 正常情况下应当存在：
# fc1.weight
# ============================================================

weight_key = "fc1.weight"

if weight_key not in state_dict:

    print("\n当前模型中包含的参数名称：")

    for key in state_dict.keys():
        print(key)

    raise KeyError(
        f"模型中没有找到参数：{weight_key}"
    )


# ============================================================
# 5. 取得BNN第一层已经训练好的权重
#
# 假设fc1结构是：
#
# 1024 → 512
#
# 则权重尺寸为：
#
# [512, 1024]
# ============================================================

first_layer_weight = state_dict[weight_key]

print(
    "\n第一层权重尺寸：",
    first_layer_weight.shape
)


# ============================================================
# 6. 选择第一层的一个神经元
#
# 第0个神经元拥有1024个权重
# ============================================================

neuron_index = 0

real_weight = first_layer_weight[
    neuron_index
].flatten()

print(
    f"当前测试第一层第{neuron_index}个神经元"
)

print(
    "该神经元权重数量：",
    real_weight.numel()
)


# ============================================================
# 7. 将训练好的浮点权重二值化
#
# weight >= 0 → +1
# weight <  0 → -1
# ============================================================

binary_weight = torch.where(
    real_weight >= 0,
    torch.ones_like(real_weight),
    -torch.ones_like(real_weight)
)

print("\n训练权重前20项：")
print(real_weight[:20])

print("\n二值权重前20项：")
print(binary_weight[:20])


# ============================================================
# 8. 自动查找一张RGB图像
#
# 保留：
# apple_x_x_x_crop.png
#
# 排除：
# depthcrop
# maskcrop
# ============================================================

rgb_files = sorted([
    file_path
    for file_path in dataset_folder.glob("*.png")
    if "_crop" in file_path.stem.lower()
    and "depth" not in file_path.stem.lower()
    and "mask" not in file_path.stem.lower()
])

if not rgb_files:
    raise FileNotFoundError(
        f"没有找到RGB图片：{dataset_folder}"
    )

rgb_path = rgb_files[0]

print("\n本次测试图片：")
print(rgb_path)


# ============================================================
# 9. 图像预处理
#
# RGB → 灰度 → 32×32 → Tensor
#
# 这里应当与训练时的输入尺寸保持一致。
# ============================================================

image_transform = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (32, 32)
    ),

    transforms.ToTensor()
])

rgb_image = Image.open(
    rgb_path
).convert("RGB")

rgb_tensor = image_transform(
    rgb_image
)

print("\nRGB原图尺寸：", rgb_image.size)

print(
    "处理后尺寸：",
    rgb_tensor.shape
)

print(
    "像素最小值：",
    rgb_tensor.min().item()
)

print(
    "像素最大值：",
    rgb_tensor.max().item()
)


# ============================================================
# 10. 将32×32图像展开为1024个输入
# ============================================================

input_values = rgb_tensor.flatten()

print(
    "展开后输入尺寸：",
    input_values.shape
)


# ============================================================
# 11. 输入二值化
#
# 当前使用图像平均像素值作为阈值：
#
# pixel >= mean → +1
# pixel <  mean → -1
# ============================================================

threshold = input_values.mean()

binary_input = torch.where(
    input_values >= threshold,
    torch.ones_like(input_values),
    -torch.ones_like(input_values)
)

print(
    "\n当前二值化阈值：",
    threshold.item()
)

print(
    "输入中+1数量：",
    torch.sum(binary_input == 1).item()
)

print(
    "输入中-1数量：",
    torch.sum(binary_input == -1).item()
)


# ============================================================
# 12. 检查输入数量和权重数量
# ============================================================

if binary_input.numel() != binary_weight.numel():

    raise ValueError(
        "\n输入与权重长度不一致："
        f"\n输入数量：{binary_input.numel()}"
        f"\n权重数量：{binary_weight.numel()}"
    )


# ============================================================
# 13. 普通二值点积
#
# 用于和XNOR-Popcount结果进行验证
#
# y = Σ(input × weight)
# ============================================================

normal_dot_product = torch.sum(
    binary_input * binary_weight
)


# ============================================================
# 14. 将+1/-1编码成硬件中的1/0
#
# +1 → 1
# -1 → 0
# ============================================================

input_bits = (
    binary_input > 0
).to(torch.int32)

weight_bits = (
    binary_weight > 0
).to(torch.int32)


# ============================================================
# 15. XNOR
#
# 输入位和权重位相同：
# XNOR = 1
#
# 输入位和权重位不同：
# XNOR = 0
# ============================================================

xnor_bits = (
    input_bits == weight_bits
).to(torch.int32)


# ============================================================
# 16. Popcount
#
# 统计XNOR结果中1的数量
# ============================================================

popcount = torch.sum(
    xnor_bits
)

number_of_bits = binary_input.numel()

different_count = (
    number_of_bits
    -
    popcount.item()
)


# ============================================================
# 17. 将Popcount恢复成BNN点积结果
#
# y = 2 × Popcount - N
# ============================================================

xnor_popcount_result = (
    2 * popcount
    -
    number_of_bits
)


# ============================================================
# 18. 输出验证结果
# ============================================================

print(
    "\n================ XNOR和Popcount验证 ================"
)

print(
    "输入总位数：",
    number_of_bits
)

print(
    "XNOR结果前20位："
)

print(
    xnor_bits[:20]
)

print(
    "相同位数量 Popcount：",
    popcount.item()
)

print(
    "不同位数量：",
    different_count
)

print(
    "普通二值点积结果：",
    normal_dot_product.item()
)

print(
    "XNOR-Popcount结果：",
    xnor_popcount_result.item()
)

print(
    "两种计算是否完全相同：",
    normal_dot_product.item()
    ==
    xnor_popcount_result.item()
)