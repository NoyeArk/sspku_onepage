# coding: UTF-8
"""
检查 NER 数据集的标签分布
"""
from collections import Counter
import sys

# 读取原始数据文件
def check_label_distribution(file_path):
    label_counter = Counter()
    total_lines = 0
    total_entities = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence_labels = []
        for line in f:
            line = line.strip()
            if not line:
                # 空行表示句子结束
                if current_sentence_labels:
                    # 统计实体
                    i = 0
                    while i < len(current_sentence_labels):
                        label = current_sentence_labels[i]
                        if label.startswith('B-'):
                            total_entities += 1
                        label_counter[label] += 1
                        i += 1
                    current_sentence_labels = []
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                label = parts[1]
                current_sentence_labels.append(label)
                total_lines += 1
    
    # 处理最后一个句子
    if current_sentence_labels:
        for label in current_sentence_labels:
            if label.startswith('B-'):
                total_entities += 1
            label_counter[label] += 1
    
    return label_counter, total_lines, total_entities

print("检查训练集标签分布...")
train_labels, train_lines, train_entities = check_label_distribution('youku/train.txt')

print(f"\n总字符数: {train_lines}")
print(f"总实体数: {train_entities}")
print(f"\n标签分布:")
for label, count in train_labels.most_common():
    percentage = count / train_lines * 100
    print(f"  {label:20s}: {count:8d} ({percentage:6.2f}%)")

print(f"\nO 标签占比: {train_labels['O'] / train_lines * 100:.2f}%")
print(f"实体标签占比: {(train_lines - train_labels['O']) / train_lines * 100:.2f}%")













