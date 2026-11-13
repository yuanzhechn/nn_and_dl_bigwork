# 基于LeNet-5的服饰分类与结构改进

在Fashion-MNIST上训练LeNet-5基线，比较深度可分离卷积、SE通道注意力和GELU激活。`data.py`下载并校验数据，`models.py`定义网络并统计参数量和MACs，`train.py`训练和评估，`analyze.py`生成图表和Top-3命中率，`predict.py`提供服饰类目推理。

## 环境与数据

```bash
conda create -n machine_learning python=3.10
conda activate machine_learning
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python data.py --root data
```

在已有的`machine_learning`环境中可直接激活。上面使用CUDA 12.4版PyTorch。CPU环境将索引地址末尾的`cu124`换为`cpu`。下载脚本使用发布方公布的MD5校验Fashion-MNIST文件。

## 复现实验

12轮主实验包含标准卷积基线、深度可分离卷积、SE、GELU以及深度可分离卷积与SE的组合。

```bash
python train.py --device auto --output-dir results --variants baseline dw se gelu dw_se --seeds 17 29 41 --split-seed 2026 --epochs 12 --batch-size 256
python analyze.py --metrics results/metrics.json
```

24轮补充实验对比标准卷积基线与GELU。

```bash
python train.py --device auto --output-dir results_24epoch --variants baseline gelu --seeds 17 29 41 --split-seed 2026 --epochs 24 --batch-size 256
python analyze.py --metrics results_24epoch/metrics.json
```

训练采用固定的分层验证划分，按验证准确率选择各次训练的权重，并在官方测试集上评估。输出包括逐轮记录`metrics.json`、汇总`summary.csv`、分析`analysis.json`、图表和权重。若CPU复测与原始CUDA混淆矩阵有差异，可在训练环境中运行`analyze.py`并加上`--device cuda`。

## 图片推理

从`metrics.json`读取按验证结果选中的权重，再对导出的服饰图像运行推理。

```bash
checkpoint=$(python -c "import json; from pathlib import Path; print(json.loads(Path('results/metrics.json').read_text(encoding='utf-8'))['application']['checkpoint'])")
python predict.py --checkpoint "results/$checkpoint" --images results/application_input.png --top-k 3
```

输入适合黑底浅色、居中的单件服饰图像。白底深色服饰图可加`--invert`。脚本输出前三候选类目和对应Softmax分数。
