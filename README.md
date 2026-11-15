# 基于LeNet-5的服饰分类与结构改进

在Fashion-MNIST上训练LeNet-5基线，比较深度可分离卷积、SE通道注意力和GELU激活。`data.py`下载并校验数据，`models.py`定义网络并统计参数量和MACs，`train.py`训练和评估，`analyze.py`生成图表和Top-3命中率，`predict.py`提供服饰类目推理。

## 服务器环境与数据

报告的主实验和补充实验均在远程服务器上运行，使用 NVIDIA A10、Conda 环境 `machine_learning`、Python 3.10.20 和 PyTorch 2.6.0+cu124（CUDA 12.4）。在克隆后的项目目录执行下列命令。

```bash
conda create -n machine_learning python=3.10.20
conda activate machine_learning
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python data.py --root data
```

已有相同依赖的 `machine_learning` 环境时，激活后即可下载数据。下载脚本按发布方公布的 MD5 校验 Fashion-MNIST 归档。

## 复现实验

复跑时选用服务器上的可用 GPU。原实验用 `CUDA_VISIBLE_DEVICES=1` 选卡，GPU 编号由当前服务器配置决定。

12轮主实验包含标准卷积基线、深度可分离卷积、SE、GELU以及深度可分离卷积与SE的组合。

```bash
python -u train.py --device cuda --output-dir results --variants baseline dw se gelu dw_se --seeds 17 29 41 --split-seed 2026 --epochs 12 --batch-size 256 --threads 4
python analyze.py --metrics results/metrics.json --device cuda
```

24轮补充实验对比标准卷积基线与GELU。

```bash
python -u train.py --device cuda --output-dir results_24epoch --variants baseline gelu --seeds 17 29 41 --split-seed 2026 --epochs 24 --batch-size 256 --threads 4
python analyze.py --metrics results_24epoch/metrics.json --device cuda
```

训练使用固定的分层划分，包含 55,000 张训练图、5,000 张验证图和 10,000 张官方测试图。训练按验证准确率选权重，分析脚本校验模型、种子和最佳轮次，并在训练所用的 CUDA 环境复测官方测试集的混淆矩阵。上述命令会在各自的输出目录生成 `metrics.json`、`summary.csv`、`analysis.json`、图表和权重。再次完整训练时，请指定新的输出目录。

## 图片推理

从`metrics.json`读取按验证结果选中的权重，再对导出的服饰图像运行推理。

```bash
checkpoint=$(python -c "import json; from pathlib import Path; print(json.loads(Path('results/metrics.json').read_text(encoding='utf-8'))['application']['checkpoint'])")
python predict.py --checkpoint "results/$checkpoint" --images results/application_input.png --top-k 3
```

输入适合黑底浅色、居中的单件服饰图像。白底深色服饰图可加`--invert`。脚本输出前三候选类目和对应Softmax分数。
