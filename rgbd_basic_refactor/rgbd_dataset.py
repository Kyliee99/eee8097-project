from pathlib import Path
from typing import Callable, Optional
import random

from PIL import Image
from torch.utils.data import Dataset


class RGBDObjectDataset(Dataset):
    """
    RGB-D Object Dataset 读取类。

    新的数据划分方法：
        1. 扫描每个类别目录下所有实例子目录中的普通 RGB PNG 图片；
        2. 对每个类别的图片进行固定随机打乱；
        3. 按图片数量划分：
               70% -> train
               15% -> val
               15% -> test

    由于每个类别分别划分，因此 train、val、test 中都会尽量保留
    相同的类别比例。

    random_seed 用来保证每次运行都得到相同的数据划分。
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        random_seed: int = 42
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.random_seed = random_seed

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split 必须是 'train'、'val' 或 'test'"
            )

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"找不到数据集目录：{self.root_dir.resolve()}"
            )

        # 获取并固定类别名称顺序
        self.class_names = sorted(
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir()
        )

        # 类别名称映射为数字标签
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.class_names)
        }

        # 每个元素保存：(图片路径, 类别标签)
        self.samples = []

        self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"{split} 数据集中没有找到 RGB 图片，请检查文件名和路径。"
            )

    @staticmethod
    def _is_rgb_image(file_path: Path) -> bool:
        """
        只选择普通 RGB 裁剪图像。

        读取：
            apple_1_1_1_crop.png

        排除：
            apple_1_1_1_depthcrop.png
            apple_1_1_1_maskcrop.png
            apple_1_1_1_loc.txt
        """
        file_name = file_path.name.lower()

        if not file_name.endswith("_crop.png"):
            return False

        if file_name.endswith("_depthcrop.png"):
            return False

        if file_name.endswith("_maskcrop.png"):
            return False

        return True

    def _split_class_images(self, image_paths):
        """
        将某一个类别的全部图片随机划分为 70% / 15% / 15%。

        同一个 random_seed、相同的文件列表和相同的排序顺序，
        会产生完全相同的 train、val、test 划分。
        """
        image_paths = sorted(image_paths)
        number_of_images = len(image_paths)

        if number_of_images < 3:
            raise RuntimeError(
                f"每个类别至少需要 3 张图片，当前只有 {number_of_images} 张。"
            )

        # 每个类别独立且可复现地随机打乱
        rng = random.Random(f"{self.random_seed}:{image_paths[0].parent.parent.name}")
        rng.shuffle(image_paths)

        train_count = int(number_of_images * 0.70)
        val_count = int(number_of_images * 0.15)

        train_end = train_count
        val_end = train_count + val_count

        if self.split == "train":
            return image_paths[:train_end]

        if self.split == "val":
            return image_paths[train_end:val_end]

        return image_paths[val_end:]

    def _collect_samples(self):
        """
        扫描每个类别下的所有实例子目录，并按图片随机划分。
        """
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            class_label = self.class_to_idx[class_name]

            # 递归查找该类别下所有普通 RGB PNG 图片
            class_image_paths = [
                file_path
                for file_path in class_dir.rglob("*.png")
                if file_path.is_file()
                and self._is_rgb_image(file_path)
            ]

            if len(class_image_paths) == 0:
                raise RuntimeError(
                    f"类别 {class_name} 中没有找到普通 RGB PNG 图片。"
                )

            selected_images = self._split_class_images(
                class_image_paths
            )

            for image_path in selected_images:
                self.samples.append(
                    (image_path, class_label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"无法读取图片：{image_path}"
            ) from error

        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":

    from collections import Counter
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Grayscale(
            num_output_channels=1
        ),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,)
        )
    ])

    dataset_root = (
        r"D:\personalproject\BNN_Project\data\rgbd-dataset_eval"
    )

    # 三个数据集必须使用相同的 random_seed，
    # 才能保证三部分来自同一套固定随机划分。
    split_seed = 42

    train_dataset = RGBDObjectDataset(
        root_dir=dataset_root,
        split="train",
        transform=transform,
        random_seed=split_seed
    )

    val_dataset = RGBDObjectDataset(
        root_dir=dataset_root,
        split="val",
        transform=transform,
        random_seed=split_seed
    )

    test_dataset = RGBDObjectDataset(
        root_dir=dataset_root,
        split="test",
        transform=transform,
        random_seed=split_seed
    )

    print("类别数量：", len(train_dataset.class_names))
    print("类别名称：", train_dataset.class_names)

    total_images = (
        len(train_dataset)
        + len(val_dataset)
        + len(test_dataset)
    )

    print("\n数据集划分")
    print("-" * 50)
    print(
        f"训练图片数量：{len(train_dataset)} "
        f"({len(train_dataset) / total_images:.2%})"
    )
    print(
        f"验证图片数量：{len(val_dataset)} "
        f"({len(val_dataset) / total_images:.2%})"
    )
    print(
        f"测试图片数量：{len(test_dataset)} "
        f"({len(test_dataset) / total_images:.2%})"
    )
    print(f"总图片数量：{total_images}")

    # 检查三个数据集之间是否有重复图片
    train_paths = {
        image_path
        for image_path, _ in train_dataset.samples
    }
    val_paths = {
        image_path
        for image_path, _ in val_dataset.samples
    }
    test_paths = {
        image_path
        for image_path, _ in test_dataset.samples
    }

    print("\n重复图片检查")
    print("-" * 50)
    print("Train 与 Val 重复：", len(train_paths & val_paths))
    print("Train 与 Test 重复：", len(train_paths & test_paths))
    print("Val 与 Test 重复：", len(val_paths & test_paths))

    # 打印每个数据集中的类别数量
    for split_name, dataset in [
        ("Train", train_dataset),
        ("Validation", val_dataset),
        ("Test", test_dataset)
    ]:
        label_counts = Counter(
            label
            for _, label in dataset.samples
        )

        print(f"\n{split_name} 每类图片数量")
        print("-" * 50)

        for class_index, class_name in enumerate(
            dataset.class_names
        ):
            print(
                f"{class_name:<15} "
                f"{label_counts[class_index]}"
            )

    image, label = train_dataset[0]

    print("\n第一张训练图片")
    print("-" * 50)
    print("图片 Shape：", image.shape)
    print("数字标签：", label)
    print(
        "类别名称：",
        train_dataset.class_names[label]
    )
    print("像素最小值：", image.min().item())
    print("像素最大值：", image.max().item())
