"""
纯 CNN + CTC：卷积提取沿宽度方向序列帧，1×1 卷积映射到类别，输出 CTC 用 log 概率（无 RNN/LSTM）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HouseNumberCNN(nn.Module):
    """输入 (N,3,32,W)，输出 log_probs (T,N,C)，T = cnn_output_length(W)。"""

    def __init__(self, num_classes: int = 11, dropout: float = 0.25) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def cnn_output_length(width: int) -> int:
        """与 self.cnn 中三次 W 方向 2 倍下采样一致（前三个 MaxPool2d(2,2)）。"""
        w = width // 2
        w = w // 2
        w = w // 2
        return max(1, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        n, c, h, w = feat.shape
        if h != 1:
            feat = F.adaptive_avg_pool2d(feat, (1, w))
        feat = self.dropout(feat)
        logits = self.head(feat)
        # (N, C, 1, T) -> (T, N, C)
        logits = logits.squeeze(2).permute(2, 0, 1).contiguous()
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs


@torch.no_grad()
def greedy_ctc_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    blank: int = 10,
) -> list[str]:
    """
    贪心 CTC 解码。

    Args:
        log_probs: (T, N, C)
        input_lengths: (N,) 每条样本有效时间步（从左起计）。
        blank: blank 类别下标。

    Returns:
        长度为 N 的预测数字串列表。
    """
    best = log_probs.argmax(dim=2)
    lens = input_lengths.cpu().tolist()
    results: list[str] = []
    for b in range(log_probs.size(1)):
        t_end = int(lens[b])
        prev = blank
        collapsed: list[int] = []
        for t in range(t_end):
            p = int(best[t, b].item())
            if p == blank:
                prev = blank
                continue
            if p != prev:
                collapsed.append(p)
            prev = p
        results.append("".join(str(d) for d in collapsed))
    return results


def sequence_accuracy(pred: list[str], gold: list[str]) -> float:
    """序列完全一致准确率。"""
    if not pred:
        return 0.0
    correct = sum(int(p == g) for p, g in zip(pred, gold))
    return correct / len(pred)
