# SVHN Format 1（Full Numbers）门牌号识别

工程根目录：`course/深度学习技术与应用/homework3/`（本目录）。

使用 **train.tar.gz** 训练、**test.tar.gz** 测试；**不使用** `extra.tar.gz`。  
推理阶段仅输入整幅 PNG，**不把** `test/digitStruct.mat` 中的 bbox 作为模型输入；评测时仍可用该 mat 中的标签计算序列准确率。

## 数据准备

1. 从 [SVHN 官网](http://ufldl.stanford.edu/housenumbers/) 下载 **Format 1**：`train.tar.gz`、`test.tar.gz`。
2. 解压到本工程下的 **`data/`**（与当前仓库一致），目录结构为：

```
data/train/   # 含 *.png 与 digitStruct.mat
data/test/
```

（若你放在 `data/svhn_format1/train` 等多一层目录，运行时加 `--data_root` 指向该父目录即可。）

## 环境

```bash
cd /Users/qileizhou/Desktop/code/sspku_onepage/course/深度学习技术与应用/homework3
pip install -r requirements.txt
```

## 训练

```bash
python train.py --epochs 20 --batch_size 32 --lr 0.001
```

（默认 `--data_root data`，即使用上面的 `data/train`、`data/test`。）

主要产物：

- `checkpoints/best.pt`：验证集序列准确率最高的权重
- `logs/loss.png`、`logs/seq_acc.png`：训练曲线

可调参数见 `python train.py --help`。

## 测试集评测（整图推理）

```bash
python eval_test.py --checkpoint checkpoints/best.pt
```

输出 **test** 集序列级准确率（预测字符串与真值完全一致的比例）。

## 方法说明（用于报告）

- **模型**：**纯 CNN**（卷积骨干 + 高度压到 1 后沿宽度为时间维 + **1×1 卷积**分类头），`CTCLoss`（`blank=10`，共 11 类：数字 0–9 + blank）；**不使用 LSTM/RNN**。
- **标签**：从 `digitStruct.mat` 读取各数字 bbox 的 `label`（SVHN 中 10 表示数字 0），按 **`left` 从小到大** 排序后拼成门牌字符串；**训练时也不把 bbox 输入网络**，仅用于生成标签序列（与测试约束一致：网络始终只见整图）。
