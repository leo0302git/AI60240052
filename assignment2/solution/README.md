# MNIST 手写数字识别

作者：钟锦程（2026210748），电子工程系。

本项目比较线性分类器、不同隐藏层宽度与学习率的 MLP、普通 CNN 和数据增强 CNN。使用固定的训练/验证划分，根据验证集准确率保存每次实验的最佳模型，最后在测试集上计算准确率和宏平均 F1。`train.py` 还生成学习曲线、混淆矩阵与自制字体数字样本图。

## 运行

建议 Python 3.11。在 `assignment2/` 目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r solution/requirements.txt
python solution/train.py
```

已有网络学堂附件时，把 `mnist.zip` 放在 `solution/` 的上一级目录即可。程序会从中读取 MNIST 的四个 IDX 文件。没有该附件时，程序会通过 `torchvision.datasets.MNIST` 下载数据。数据缓存在上一级 `data/` 中；首次运行需要下载或解压，后续运行直接读取缓存。

输出在 `results/`：`metrics.json` 是逐次实验的真实数据；`learning_curves.png`、`confusion_matrix.png` 和 `font_digits.png` 用于报告。`checkpoints/` 保存各模型在验证集上最好的权重。原始数据与权重均无需上传到代码仓库，运行命令可以重新生成它们。

默认自动选择 Apple MPS 或 CPU；也可指定 `--device cpu`。`--quick` 只用于验证运行流程，使用少量数据训练一轮，不会得到报告中的指标。正式复现实验请使用上面的完整命令。各模型每轮最多训练 6 个 epoch，若验证准确率连续两轮未提高则提前停止。随机种子固定为 2026，但不同硬件和 PyTorch 版本的浮点运算可能产生少量差异。
