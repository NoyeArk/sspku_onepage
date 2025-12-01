# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = "[PAD]", "[CLS]"  # padding符号, bert中综合信息符号


def build_dataset(config):
    def load_dataset(path, pad_size=128):
        """
        加载 NER 数据集（BIO 格式）

        数据格式：每行是 "字符\t标签"，空行表示句子结束
        """
        contents = []
        sentences = []
        labels = []

        with open(path, "r", encoding="UTF-8") as f:
            current_sentence = []
            current_labels = []

            for line in tqdm(f):
                line = line.strip()
                if not line:
                    # 空行表示句子结束
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                    continue

                # 解析：字符\t标签
                parts = line.split()
                if len(parts) < 2:
                    continue

                char = parts[0]
                label = parts[1]

                current_sentence.append(char)
                current_labels.append(label)

            # 处理最后一个句子（如果文件末尾没有空行）
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)

        print(f"加载了 {len(sentences)} 个句子")

        # 处理每个句子
        for sentence, label_seq in zip(sentences, labels):
            # 将句子转换为字符串
            text = "".join(sentence)

            # 使用 tokenizer 分词
            tokens = config.tokenizer.tokenize(text)
            tokens = [CLS] + tokens  # 添加 [CLS] token

            # 将标签映射到 id
            # BERT tokenizer 对中文的处理：通常每个中文字符对应一个 token
            # 对于英文/数字，可能会分割成多个 subword
            label_ids = [config.label2id["O"]]  # CLS token 的标签

            char_idx = 0
            for token in tokens[1:]:  # 跳过 CLS
                if token.startswith("##"):
                    # 子词（subword），使用前一个标签（保持 I- 标签的连续性）
                    if label_ids:
                        prev_label_id = label_ids[-1]
                        # 如果前一个标签是 B- 或 I-，保持 I-；否则用 O
                        prev_label = config.id2label[prev_label_id]
                        if prev_label.startswith("B-") or prev_label.startswith("I-"):
                            # 保持实体类型，但改为 I-
                            entity_type = (
                                prev_label.split("-")[-1] if "-" in prev_label else ""
                            )
                            if entity_type:
                                label_ids.append(
                                    config.label2id.get(
                                        f"I-{entity_type}", config.label2id["O"]
                                    )
                                )
                            else:
                                label_ids.append(prev_label_id)
                        else:
                            label_ids.append(prev_label_id)
                    else:
                        label_ids.append(config.label2id["O"])
                else:
                    # 新词，获取对应字符的标签
                    if char_idx < len(label_seq):
                        label = label_seq[char_idx]
                        label_ids.append(
                            config.label2id.get(label, config.label2id["O"])
                        )
                        char_idx += 1
                    else:
                        label_ids.append(config.label2id["O"])

            # 转换为 token ids
            token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
            seq_len = len(token_ids)

            # Padding
            mask = []
            if pad_size:
                if len(token_ids) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token_ids))
                    token_ids += [0] * (pad_size - len(token_ids))
                    label_ids += [config.label2id["O"]] * (pad_size - len(label_ids))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    label_ids = label_ids[:pad_size]
                    seq_len = pad_size

            contents.append((token_ids, label_ids, seq_len, mask))

        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        """
        转换为 tensor

        Args:
            datas: list of (token_ids, label_ids, seq_len, mask)

        Returns:
            (input_ids, seq_len, mask), labels
        """
        input_ids = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        labels = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (input_ids, seq_len, mask), labels

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                self.index * self.batch_size : (self.index + 1) * self.batch_size
            ]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
