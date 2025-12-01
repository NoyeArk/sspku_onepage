# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
import os


class Config(object):
    """配置参数 - 支持文本分类和 NER 两种任务"""

    def __init__(self, dataset):
        # 自动检测任务类型：如果路径包含 /data/ 则是分类任务，否则是 NER 任务
        if "/data/" in dataset or os.path.exists(dataset + "/data/train.txt"):
            self.task_type = "classification"
            self.model_name = "bert"
            self.train_path = dataset + "/data/train.txt"  # 训练集
            self.dev_path = dataset + "/data/dev.txt"  # 验证集
            self.test_path = dataset + "/data/test.txt"  # 测试集
            self.class_list = [
                x.strip() for x in open(dataset + "/data/class.txt").readlines()
            ]  # 类别名单
            self.num_classes = len(self.class_list)  # 类别数
            self.batch_size = 128  # mini-batch大小
            self.pad_size = 32  # 每句话处理成的长度(短填长切)
            self.rnn_hidden = 256  # LSTM 隐藏层大小
        else:
            # NER 任务
            self.task_type = "ner"
            self.model_name = "bert_rcnn_ner"
            self.train_path = dataset + "/train.txt"  # 训练集
            self.dev_path = dataset + "/dev.txt"  # 验证集
            self.test_path = dataset + "/test.txt"  # 测试集

            # NER 标签列表（BIO 格式）
            self.label_list = [
                "O",
                "B-PER",
                "I-PER",
                "B-TELEVISION",
                "I-TELEVISION",
                "B-MISC",
                "I-MISC",
            ]
            self.num_labels = len(self.label_list)  # 标签数量
            self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
            self.id2label = {idx: label for idx, label in enumerate(self.label_list)}
            self.num_classes = self.num_labels  # 为了兼容性
            self.batch_size = 16  # mini-batch大小（NER任务通常用较小的batch）
            self.pad_size = 128  # 每句话处理成的长度(短填长切)
            self.rnn_hidden = 256  # LSTM 隐藏层大小

        self.save_path = (
            dataset + "/saved_dict/" + self.model_name + ".ckpt"
        )  # 模型训练结果
        self.device = torch.device(
            "cuda:1" if torch.cuda.is_available() else "cpu"
        )  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 3  # epoch数
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = "./bert_pretrain"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.dropout = 0.1
        self.num_layers = 2  # LSTM 层数


class Model(nn.Module):
    """
    BERT + RCNN 模型 - 支持文本分类和 NER 两种任务
    
    RCNN (Recurrent Convolutional Neural Network) 架构：
    1. BERT 获取词嵌入
    2. LSTM 处理序列信息
    3. 拼接 BERT 输出和 LSTM 输出（融合上下文信息）
    4. 根据任务类型输出不同结果
    """

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.task_type = config.task_type
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM 层：处理序列信息
        self.lstm = nn.LSTM(
            config.hidden_size,  # 输入维度：BERT 的 hidden_size
            config.rnn_hidden,  # LSTM 隐藏层大小
            config.num_layers,  # LSTM 层数
            bidirectional=True,  # 双向 LSTM
            batch_first=True,  # 输入格式为 (batch, seq, feature)
            dropout=(
                config.dropout if config.num_layers > 1 else 0
            ),  # 多层时使用 dropout
        )

        if self.task_type == "classification":
            # 文本分类任务：使用 MaxPooling 池化整个序列
            self.maxpool = nn.MaxPool1d(config.pad_size)
            self.fc = nn.Linear(
                config.rnn_hidden * 2 + config.hidden_size, config.num_classes
            )
        else:
            # NER 任务：对每个位置的拼接特征进行分类
            self.dropout = nn.Dropout(config.dropout)
            self.classifier = nn.Linear(
                config.hidden_size + config.rnn_hidden * 2, config.num_labels
            )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (input_ids, seq_len, mask)
                - input_ids: [batch_size, seq_len]
                - seq_len: [batch_size]
                - mask: [batch_size, seq_len]

        Returns:
            classification: [batch_size, num_classes]
            ner: [batch_size, seq_len, num_labels]
        """
        input_ids = x[0]  # [batch_size, seq_len]
        attention_mask = x[2]  # [batch_size, seq_len]

        # BERT 输出
        encoder_out, text_cls = self.bert(
            input_ids, attention_mask=attention_mask, output_all_encoded_layers=False
        )
        # encoder_out: [batch_size, seq_len, hidden_size]

        # LSTM 处理序列
        lstm_out, _ = self.lstm(encoder_out)
        # lstm_out: [batch_size, seq_len, rnn_hidden * 2] (双向)

        # RCNN 核心：拼接 BERT 输出和 LSTM 输出
        rcnn_out = torch.cat((encoder_out, lstm_out), 2)
        # rcnn_out: [batch_size, seq_len, hidden_size + rnn_hidden * 2]

        rcnn_out = F.relu(rcnn_out)  # 激活函数

        if self.task_type == "classification":
            # 文本分类任务：使用 MaxPooling 池化整个序列
            rcnn_out = rcnn_out.permute(0, 2, 1)  # [batch_size, feature, seq_len]
            out = self.maxpool(rcnn_out).squeeze()  # [batch_size, feature]
            out = self.fc(out)  # [batch_size, num_classes]
            return out
        else:
            # NER 任务：对每个位置的拼接特征进行分类
            rcnn_out = self.dropout(rcnn_out)
            logits = self.classifier(rcnn_out)  # [batch_size, seq_len, num_labels]
            return logits
