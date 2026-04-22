"""
SVHN Format 1 整图数据集：图像 + 门牌字符串；CTC 用整数标签序列。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from digit_struct import parse_digit_struct_mat
from model import HouseNumberCNN


def house_string_to_indices(s: str) -> List[int]:
    """门牌字符串 -> CTC 目标下标列表（不含 blank）。"""
    out: List[int] = []
    for ch in s:
        if not ch.isdigit():
            continue
        out.append(int(ch))
    return out


class SVHNFullImageDataset(Dataset):
    """
    Args:
        image_dir: 含 png 的目录（train 或 test）。
        mat_path: 对应 digitStruct.mat（训练用于标签；测试集若仅推理可不传 mat，本作业 eval 仍用 mat 只读标签）。
    """

    def __init__(self, image_dir: str | Path, mat_path: str | Path) -> None:
        self.image_dir = Path(image_dir)
        entries = parse_digit_struct_mat(mat_path)
        self.samples: List[Tuple[Path, str, List[int]]] = []
        for e in entries:
            p = self.image_dir / e.filename
            if not p.is_file():
                continue
            idxs = house_string_to_indices(e.house_number)
            if len(idxs) == 0:
                continue
            self.samples.append((p, e.house_number, idxs))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        path, house_str, label_idxs = self.samples[i]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # 统一高度到 32，宽度按比例缩放
        target_h = 32
        new_w = max(1, int(round(w * target_h / float(h))))
        img = img.resize((new_w, target_h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # HWC -> CHW
        chw = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(chw)
        y = torch.tensor(label_idxs, dtype=torch.long)
        return x, y, new_w, house_str


def _collate_ctc(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int, str]],
) -> dict:
    """按 batch 内最大宽 pad 图像；拼 target 与长度。"""
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    orig_widths = [b[2] for b in batch]
    houses = [b[3] for b in batch]

    max_w = max(orig_widths)
    padded = []
    for x in xs:
        _, h, w = x.shape
        pad_w = max_w - w
        if pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, 0), value=0.0)
        padded.append(x)
    images = torch.stack(padded, dim=0)

    target_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    targets = torch.cat(ys, dim=0) if sum(len(y) for y in ys) > 0 else torch.zeros(0, dtype=torch.long)

    time_steps = torch.tensor(
        [HouseNumberCNN.cnn_output_length(w) for w in orig_widths], dtype=torch.long
    )

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lens,
        "input_lengths": time_steps,
        "houses": houses,
    }


def build_train_val_loaders(
    data_root: Path,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    在官方 train 上划分 train/val；均使用 train/digitStruct.mat 生成标签。
    """
    data_root = Path(data_root)
    train_dir = data_root / "train"
    mat_path = train_dir / "digitStruct.mat"
    full_ds = SVHNFullImageDataset(train_dir, mat_path)
    n = len(full_ds)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = int(round(n * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds: Dataset = Subset(full_ds, train_idx)
    val_ds: Dataset = Subset(full_ds, val_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_ctc,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_ctc,
        pin_memory=pin,
    )
    return train_loader, val_loader


def build_test_loader(
    data_root: Path, batch_size: int, num_workers: int = 0
) -> DataLoader:
    data_root = Path(data_root)
    test_dir = data_root / "test"
    mat_path = test_dir / "digitStruct.mat"
    ds = SVHNFullImageDataset(test_dir, mat_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_ctc,
        pin_memory=torch.cuda.is_available(),
    )
