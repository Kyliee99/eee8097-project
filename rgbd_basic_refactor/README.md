# RGB-D Object Dataset：基础架构重构版

这个版本只保留 **5 个核心 Python 文件**，不再使用多层目录和大量模板文件。

## 结构

```text
RGB-D_Object_Dataset/
├── rgbd_dataset.py        # 你原来的数据集文件
├── common.py              # 公共数据加载、预处理和分类指标
├── torch_pipeline.py      # MLP / BNN 共用训练与测试流程
├── mlp.py                 # MLP 算法与运行入口
├── bnn.py                 # BNN、XNOR/Popcount 与运行入口
└── tsetlin.py             # Tsetlin 算法与运行入口
```

## 文件职责

### common.py

- 随机种子
- CPU / GPU
- RGB 转灰度、32×32、Normalize
- train / val / test Dataset 和 DataLoader
- Accuracy、Precision、Recall、F1、分类报告、混淆矩阵

### torch_pipeline.py

MLP 和 BNN 共用：

- 训练循环
- 验证集评估
- Early Stopping
- 保存和加载最佳 `.pth`
- 测试集评估
- CSV 训练历史

### mlp.py

只保留：

- MLP 网络结构
- MLP 参数
- MLP 优化器
- 调用公共训练流程

### bnn.py

只保留：

- SignSTE
- BinaryLinear
- XNORPopcountLinear
- BNN 网络结构
- 二值权重裁剪
- 调用公共训练流程

### tsetlin.py

只保留 Tsetlin 特有内容：

- Boolean 阈值编码
- DataLoader 转 NumPy
- TMClassifier
- Tsetlin 训练、验证、测试
- `.pkl` 和 JSON 保存

## 使用

把 5 个 Python 文件复制到 `rgbd_dataset.py` 所在目录，然后分别运行：

```powershell
python mlp.py
python bnn.py
python tsetlin.py
```

## 数据路径

默认数据目录：

```text
BNN_Project/data/rgbd-dataset_eval
```

需要修改时只修改 `common.py` 中：

```python
DATASET_ROOT = PROJECT_ROOT / "data" / "rgbd-dataset_eval"
```
