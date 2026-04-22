"""
解析 SVHN Format 1 的 digitStruct.mat（通常为 MATLAB v7.3 / HDF5）。

从每条记录读取文件名与各数字的 label、left，用于按阅读顺序生成门牌字符串。
SVHN 约定：label 为 1–9 表示数字 1–9，label 为 10 表示数字 0。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import h5py
import numpy as np


@dataclass
class DigitStructEntry:
    """单张图片：文件名与按 left 排序后的门牌数字串。"""

    filename: str
    house_number: str


def _deref_ints(f: h5py.File, bbox: h5py.Group, key: str) -> List[int]:
    """读取 bbox 子组中某一字段，统一为 int 列表（支持单数字与多数字）。"""
    if key not in bbox:
        return []
    node = bbox[key]
    if not isinstance(node, h5py.Dataset):
        return []
    raw = node[()]
    out: List[int] = []

    # HDF5 中多数字常为 object dtype，元素为 Reference
    if raw.dtype == object:
        flat = np.ravel(raw)
        for ref in flat:
            if ref is None:
                continue
            try:
                val = f[ref][()]
            except (TypeError, ValueError, KeyError):
                continue
            out.append(int(np.asarray(val).squeeze()))
        return out

    # 普通数值数组或标量
    arr = np.asarray(raw).squeeze()
    if arr.ndim == 0:
        return [int(arr)]
    return [int(x) for x in np.ravel(arr)]


def _read_image_name(f: h5py.File, name_ds: h5py.Dataset, index: int) -> str:
    """digitStruct/name 中第 index 条对应的 png 文件名。"""
    if name_ds.ndim >= 2:
        ref = name_ds[index, 0]
    else:
        ref = name_ds[index]
    if isinstance(ref, np.ndarray) and ref.shape:
        ref = ref.flat[0]
    chars = f[ref][()]
    chars = np.asarray(chars).squeeze()
    if chars.ndim == 0:
        return chr(int(chars))
    return "".join(chr(int(c)) for c in np.ravel(chars))


def svhn_label_to_digit_char(label: int) -> str:
    """SVHN label -> 字符。10 表示 '0'。"""
    v = int(label)
    if v == 10:
        return "0"
    if 1 <= v <= 9:
        return str(v)
    # 极少数异常值：尽量容错
    if v == 0:
        return "0"
    return str(v % 10)


def parse_digit_struct_mat(mat_path: str | Path) -> List[DigitStructEntry]:
    """
    解析 digitStruct.mat，返回与 mat 中顺序一致的条目列表。

    Args:
        mat_path: digitStruct.mat 路径。

    Returns:
        每条含 filename 与 house_number（按 left 排序后的数字串）。
    """
    mat_path = Path(mat_path)
    entries: List[DigitStructEntry] = []

    with h5py.File(mat_path, "r") as f:
        ds = f["digitStruct"]
        name_ds = ds["name"]
        bbox_ds = ds["bbox"]
        n = int(name_ds.shape[0])

        for i in range(n):
            filename = _read_image_name(f, name_ds, i)
            if bbox_ds.ndim >= 2:
                bb_ref = bbox_ds[i, 0]
            else:
                bb_ref = bbox_ds[i]
            if isinstance(bb_ref, np.ndarray) and bb_ref.size:
                bb_ref = bb_ref.flat[0]
            bbox = f[bb_ref]

            labels = _deref_ints(f, bbox, "label")
            lefts = _deref_ints(f, bbox, "left")

            if len(labels) == 0:
                house = ""
            elif len(lefts) == len(labels):
                pairs = sorted(zip(lefts, labels), key=lambda x: x[0])
                house = "".join(svhn_label_to_digit_char(lb) for _, lb in pairs)
            else:
                # left 缺失时退化为 mat 内顺序
                house = "".join(svhn_label_to_digit_char(lb) for lb in labels)

            entries.append(DigitStructEntry(filename=filename, house_number=house))

    return entries
