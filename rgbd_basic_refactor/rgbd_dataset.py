from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class RGBDObjectDataset(Dataset):
    """
    RGB-D Object Dataset读取类。

    数据结构示例：
    root/
        apple/
            apple_1/
            apple_2/
            ...
        binder/
            binder_1/
            ...

    split:
        "train" -> 除最后两个实例外的所有实例
        "val"   -> 倒数第二个实例
        "test"  -> 最后一个实例
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split必须是 'train'、'val' 或 'test'"
            )

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"找不到数据集目录：{self.root_dir.resolve()}"
            )

        # 获取类别名称
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

        # 保存所有样本：
        # 每一个元素是 (图片路径, 类别标签)
        self.samples = []

        self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"{split}数据集中没有找到RGB图片，请检查文件名和路径。"
            )

    @staticmethod
    def _is_rgb_image(file_path: Path) -> bool:
        """
        只选择普通RGB裁剪图像。

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

    def _select_instances(self, instance_dirs):
        """
        根据split选择实例文件夹。
        """

        instance_dirs = sorted(
            instance_dirs,
            key=lambda path: path.name
        )

        number_of_instances = len(instance_dirs)

        if number_of_instances < 3:
            raise RuntimeError(
                f"类别至少需要3个实例，当前只有{number_of_instances}个。"
            )

        if self.split == "train":
            # 除最后两个实例外，其余用于训练
            return instance_dirs[:-2]

        if self.split == "val":
            # 倒数第二个用于验证
            return [instance_dirs[-2]]

        # 最后一个用于测试
        return [instance_dirs[-1]]

    def _collect_samples(self):
        """
        扫描所有类别与实例文件夹。
        """

        for class_name in self.class_names:

            class_dir = self.root_dir / class_name
            class_label = self.class_to_idx[class_name]

            instance_dirs = [
                folder
                for folder in class_dir.iterdir()
                if folder.is_dir()
            ]

            selected_instances = self._select_instances(
                instance_dirs
            )

            for instance_dir in selected_instances:

                # 递归扫描实例目录中的所有文件
                for file_path in instance_dir.rglob("*"):

                    if (
                        file_path.is_file()
                        and self._is_rgb_image(file_path)
                    ):
                        self.samples.append(
                            (file_path, class_label)
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

    print("类别数量：", len(train_dataset.class_names))
    print("类别名称：", train_dataset.class_names)

    print("训练图片数量：", len(train_dataset))
    print("验证图片数量：", len(val_dataset))
    print("测试图片数量：", len(test_dataset))

    image, label = train_dataset[0]

    print("第一张图片Shape：", image.shape)
    print("第一张图片标签：", label)
    print(
        "第一张图片类别：",
        train_dataset.class_names[label]
    )

    print("像素最小值：", image.min().item())
    print("像素最大值：", image.max().item())