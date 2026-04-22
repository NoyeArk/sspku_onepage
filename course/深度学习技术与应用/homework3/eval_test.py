"""
在官方 test 集上评测：仅整图前向；digitStruct.mat 仅用于读取真值标签算准确率。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import build_test_loader
from model import HouseNumberCNN, greedy_ctc_decode, sequence_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVHN Format1 测试集序列准确率")
    p.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="含 train/、test/ 的数据根路径（默认 data；评测只用 test/）",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="train.py 保存的权重路径",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_root = (root / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    ckpt_path = (root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    device = torch.device(args.device)

    loader = build_test_loader(
        data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model = HouseNumberCNN(num_classes=11).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_pred: list[str] = []
    all_gold: list[str] = []

    for batch in tqdm(loader, desc="test"):
        images = batch["images"].to(device, non_blocking=True)
        il = batch["input_lengths"]
        houses: list[str] = batch["houses"]

        log_probs = model(images)
        pred = greedy_ctc_decode(log_probs.cpu(), il, blank=10)
        all_pred.extend(pred)
        all_gold.extend(houses)

    acc = sequence_accuracy(all_pred, all_gold)
    print(f"test 序列准确率（整图推理）: {acc:.4f}")


if __name__ == "__main__":
    main()
