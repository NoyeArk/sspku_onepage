"""
设备管理工具 - 自动检测可用设备 (CUDA / MPS / CPU)
macOS Apple Silicon 兼容
"""
import torch
from torch import cuda

# 模块级单例，所有文件共享同一个 device
if cuda.is_available():
    _device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    _device = torch.device("mps")
else:
    _device = torch.device("cpu")


def get_device():
    """返回当前可用的计算设备"""
    return _device


def to_device(tensor):
    """将张量移动到当前设备（兼容 Variable 和 Tensor）"""
    return tensor.to(_device)
