---
name: Code Generation Model Implementation
overview: 基于CONCODE数据集实现代码生成深度学习模型，包括数据预处理、Seq2Seq模型训练、评估，以及PPT汇报材料
todos:
  - id: create-preprocess
    content: 创建数据预处理脚本 preprocess.py
    status: completed
  - id: create-model
    content: 创建 Transformer Seq2Seq 模型 model.py
    status: completed
  - id: create-train
    content: 创建训练脚本 train.py
    status: completed
    dependencies:
      - create-preprocess
      - create-model
  - id: create-translate
    content: 创建推理脚本 translate.py
    status: completed
    dependencies:
      - create-model
  - id: create-main
    content: 创建主运行脚本 main.py
    status: completed
    dependencies:
      - create-preprocess
      - create-model
      - create-train
      - create-translate
  - id: create-pptx
    content: 使用 [skill:pptx] 创建 PPT 汇报材料 presentation.pptx
    status: completed
---

## 用户需求

在 `code` 目录中完成一个完整的 Code Generation 代码生成系统，基于 CONCODE 数据集训练深度神经网络模型。

## 核心功能

1. **数据预处理**：读取 JSONL 格式的 CONCODE 数据集，进行 NL 和 Code 的分词与向量化
2. **模型实现**：基于 Transformer 的 Seq2Seq 模型，包含编码器（NL理解）和解码器（代码生成）
3. **模型训练**：使用教师强制（Teacher Forcing）策略训练，支持GPU/CPU
4. **代码推理**：使用 Beam Search 进行代码生成
5. **模型评估**：计算 Exact Match 和 BLEU 分数
6. **PPT汇报**：包含模型结构图、评估结果、训练曲线

## 技术选型

- **框架**：PyTorch
- **模型**：Transformer (Encoder-Decoder)
- **评估指标**：Exact Match、BLEU
- **PPT生成**：python-pptx

## 目录结构

```
code/
├── evaluate.py      # [已存在] 评估脚本
├── dataset.py       # [新建] 数据集类 + 词汇表
├── model.py         # [新建] Transformer Seq2Seq 模型
├── train.py         # [新建] 训练脚本（含推理和评估）
└── presentation.pptx # [新建] PPT汇报材料
```

## 模型方案

从零实现的 Transformer Seq2Seq 模型：

- **Encoder**：处理自然语言描述（NL）
- **Decoder**：生成代码 tokens
- **注意力机制**：Multi-Head Attention
- **自定义词汇表**：基于训练数据构建

## 技术栈

- **深度学习框架**：PyTorch
- **分词工具**：nltk / re
- **评估指标**：BLEU (nltk)、Exact Match
- **PPT生成**：python-pptx

## 实现方案

### 1. 数据集类 (dataset.py)

- 读取JSONL数据，构建词汇表（统计词频）
- NL和Code分别tokenize（空格分词）
- 特殊token：PAD, BOS, EOS, UNK
- 处理变长序列，padding到固定长度
- 返回PyTorch Tensor

### 2. 模型定义 (model.py)

- **Encoder**：多层 Transformer Encoder，理解自然语言描述
- **Decoder**：多层 Transformer Decoder，生成代码token
- **注意力机制**：Multi-Head Self-Attention + Cross-Attention
- **Embedding**：词嵌入 + 位置编码

### 3. 训练脚本 (train.py)

- 训练模式：教师强制训练
- 推理模式：Beam Search 解码
- 评估模式：计算 Exact Match 和 BLEU
- 保存 learning curve

### 5. 评估

- 使用官方评估脚本 `../evaluator/evaluator.py`
- Exact Match：生成代码与参考答案完全匹配的比例（按token比较）
- BLEU：基于token分词的BLEU分数
- 评估格式：
- 答案文件：JSON格式（每行包含 `{"code": "..."}`）
- 预测文件：TXT格式（每行一个预测代码，空格分词）

## 数据路径

- 训练集：`../data/train.jsonl`（100,000条）
- 验证集：`../data/dev.jsonl`（2,000条）
- 测试集：`../data/test.jsonl`（2,000条）

## 注意事项

- 训练数据量较大（10万条），需要合理设置batch_size和epoch
- 设备自适应：优先使用 CUDA/MPS，回退到 CPU
- 代码生成任务中变量名处理（arg0, arg1等）

## Agent Extensions

### [skill:pptx]

- **用途**：生成 PPT 汇报材料
- **预期结果**：创建包含模型结构、评估结果、训练曲线的演示文稿