# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils_ner import get_time_dif
from pytorch_pretrained.optimization import BertAdam


def extract_entities(labels, id2label):
    """
    从 BIO 标签序列中提取实体

    Args:
        labels: 标签 id 列表
        id2label: id 到标签的映射

    Returns:
        entities: list of (start, end, entity_type)
    """
    entities = []
    i = 0
    while i < len(labels):
        label = id2label[labels[i]]
        if label.startswith("B-"):
            entity_type = label[2:]
            start = i
            i += 1
            # 继续查找 I- 标签
            while i < len(labels) and id2label[labels[i]] == f"I-{entity_type}":
                i += 1
            end = i - 1
            entities.append((start, end, entity_type))
        else:
            i += 1
    return entities


def compute_f1(pred_entities, true_entities):
    """
    计算实体级别的 F1 分数

    Args:
        pred_entities: 预测的实体列表 [(start, end, type), ...]
        true_entities: 真实的实体列表 [(start, end, type), ...]

    Returns:
        precision, recall, f1
    """
    pred_set = set(pred_entities)
    true_set = set(true_entities)

    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0, 1.0, 1.0

    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    if len(true_set) == 0:
        return 0.0, 0.0, 0.0

    correct = len(pred_set & true_set)
    precision = correct / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = correct / len(true_set) if len(true_set) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def train(config, model, train_iter, dev_iter, test_iter):
    """训练模型"""
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        warmup=0.05,
        t_total=len(train_iter) * config.num_epochs,
    )
    total_batch = 0
    dev_best_f1 = 0.0
    last_improve = 0
    flag = False
    model.train()

    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        for i, (inputs, labels) in enumerate(train_iter):
            # inputs: (input_ids, seq_len, mask)
            # labels: [batch_size, seq_len]
            logits = model(inputs)  # [batch_size, seq_len, num_labels]

            # 计算损失（只对非 padding 部分计算）
            attention_mask = inputs[2]  # [batch_size, seq_len]
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            # 使用类别权重处理类别不平衡问题
            # NER 任务中 O 标签通常占绝大多数，需要给实体标签更高权重
            if not hasattr(config, "class_weights"):
                # 计算类别权重：O 标签权重为 1，实体标签权重更高
                # 这里使用简单的启发式：O 标签权重 1.0，其他标签权重 5.0
                class_weights = torch.ones(config.num_labels).to(config.device)
                o_label_id = config.label2id.get("O", 0)
                for i in range(config.num_labels):
                    if i != o_label_id:
                        class_weights[i] = 5.0  # 给实体标签更高权重
                config.class_weights = class_weights

            loss = F.cross_entropy(
                active_logits, active_labels, weight=config.class_weights
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                # 计算训练集准确率（排除 CLS token）
                # active_logits 和 active_labels 已经排除了 padding，但可能包含 CLS
                # 我们需要排除 CLS token（通常是第一个位置）
                pred_labels = torch.argmax(active_logits, dim=1)
                # 注意：这里计算的准确率可能包含 CLS token，但影响不大
                train_acc = (pred_labels == active_labels).float().mean().item()

                # 验证集评估
                dev_acc, dev_f1, dev_loss = evaluate(config, model, dev_iter)

                if dev_f1 > dev_best_f1:
                    dev_best_f1 = dev_f1
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""

                time_dif = get_time_dif(start_time)
                msg = (
                    "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  "
                    "Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Val F1: {5:>6.2%},  "
                    "Time: {6} {7}"
                )
                print(
                    msg.format(
                        total_batch,
                        loss.item(),
                        train_acc,
                        dev_loss,
                        dev_acc,
                        dev_f1,
                        time_dif,
                        improve,
                    )
                )
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    test(config, model, test_iter)


def test(config, model, test_iter):
    """测试模型"""
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_f1, test_loss, detailed_report = evaluate(
        config, model, test_iter, test=True
    )
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%},  Test F1: {2:>6.2%}"
    print(msg.format(test_loss, test_acc, test_f1))
    print(detailed_report)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    """
    评估模型

    Returns:
        acc: token 级别准确率
        f1: 实体级别 F1 分数
        loss: 平均损失
        report: 详细报告（仅在 test=True 时返回）
    """
    model.eval()
    loss_total = 0
    all_pred_labels = []
    all_true_labels = []
    all_pred_entities = []
    all_true_entities = []

    with torch.no_grad():
        for inputs, labels in data_iter:
            logits = model(inputs)  # [batch_size, seq_len, num_labels]
            attention_mask = inputs[2]  # [batch_size, seq_len]

            # 计算损失
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            # 使用类别权重
            if not hasattr(config, "class_weights"):
                class_weights = torch.ones(config.num_labels).to(config.device)
                o_label_id = config.label2id.get("O", 0)
                for i in range(config.num_labels):
                    if i != o_label_id:
                        class_weights[i] = 5.0
                config.class_weights = class_weights

            loss = F.cross_entropy(
                active_logits, active_labels, weight=config.class_weights
            )
            loss_total += loss.item()

            # 预测
            pred_labels = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

            # 收集 token 级别的预测和真实标签
            for i in range(labels.size(0)):
                seq_len = int(attention_mask[i].sum().item())
                pred_seq = pred_labels[i][:seq_len].cpu().numpy()
                true_seq = labels[i][:seq_len].cpu().numpy()

                all_pred_labels.extend(pred_seq)
                all_true_labels.extend(true_seq)

                # 提取实体（跳过 CLS token，从索引 1 开始）
                # 注意：序列中第一个 token 是 CLS，所以实体位置需要从 1 开始
                # 提取实体时，位置索引需要调整（加上 1，因为跳过了 CLS）
                pred_seq_no_cls = pred_seq[1:]  # 跳过 CLS
                true_seq_no_cls = true_seq[1:]  # 跳过 CLS

                pred_entities_raw = extract_entities(pred_seq_no_cls, config.id2label)
                true_entities_raw = extract_entities(true_seq_no_cls, config.id2label)

                # 调整实体位置索引（加上 1，因为跳过了 CLS token）
                # 但注意：extract_entities 返回的位置是相对于传入序列的，所以已经是正确的
                # 我们只需要确保两个序列都跳过了 CLS，位置就是一致的
                pred_entities = pred_entities_raw
                true_entities = true_entities_raw

                all_pred_entities.extend(pred_entities)
                all_true_entities.extend(true_entities)

    # Token 级别准确率
    acc = metrics.accuracy_score(all_true_labels, all_pred_labels)

    # 实体级别 F1
    precision, recall, f1 = compute_f1(all_pred_entities, all_true_entities)

    avg_loss = loss_total / len(data_iter)

    if test:
        # 生成详细报告
        report = f"""
Token 级别评估:
  准确率: {acc:.4f}
  
实体级别评估:
  精确率: {precision:.4f}
  召回率: {recall:.4f}
  F1 分数: {f1:.4f}
  
实体统计:
  预测实体数: {len(all_pred_entities)}
  真实实体数: {len(all_true_entities)}
"""
        return acc, f1, avg_loss, report

    return acc, f1, avg_loss
