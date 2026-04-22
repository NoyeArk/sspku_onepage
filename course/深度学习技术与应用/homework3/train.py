"""
在 SVHN Format 1 官方 train 上训练 CNN+CTC，划分验证集并保存曲线与最优权重。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import amp
from tqdm import tqdm

from dataset import build_train_val_loaders
from model import HouseNumberCNN, greedy_ctc_decode, sequence_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVHN Format1 CNN+CTC 训练")
    p.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="含 train/、test/ 子目录的数据根路径（默认 data）",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="启用 CUDA 混合精度")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def plot_curve(
    xs: list[int],
    train_vals: list[float],
    val_vals: list[float],
    ylabel: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, train_vals, label="train")
    plt.plot(xs, val_vals, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.CTCLoss,
    device: torch.device,
    scaler: amp.GradScaler | None,
    train_mode: bool,
) -> tuple[float, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0
    all_pred: list[str] = []
    all_gold: list[str] = []

    for batch in tqdm(loader, leave=False):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        tl = batch["target_lengths"].to(device, non_blocking=True)
        il = batch["input_lengths"].to(device, non_blocking=True)
        houses: list[str] = batch["houses"]

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            if scaler is not None and train_mode:
                with amp.autocast("cuda", enabled=True):
                    log_probs = model(images)
                    loss = criterion(log_probs, targets, il, tl)
                if train_mode:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                log_probs = model(images)
                loss = criterion(log_probs, targets, il, tl)
                if train_mode:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1

        with torch.no_grad():
            pred = greedy_ctc_decode(log_probs.detach(), il.cpu(), blank=10)
            all_pred.extend(pred)
            all_gold.extend(houses)

    acc = sequence_accuracy(all_pred, all_gold)
    avg_loss = total_loss / max(1, total_batches)
    return avg_loss, acc


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_root = (root / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    train_loader, val_loader = build_train_val_loaders(
        data_root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = HouseNumberCNN(num_classes=11).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CTCLoss(blank=10, zero_infinity=True)
    scaler: amp.GradScaler | None = (
        amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    )

    ckpt_dir = root / "checkpoints"
    log_dir = root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    hist_train_loss: list[float] = []
    hist_val_loss: list[float] = []
    hist_train_acc: list[float] = []
    hist_val_acc: list[float] = []

    best_val_acc = -1.0
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            train_mode=True,
        )
        va_loss, va_acc = run_epoch(
            model,
            val_loader,
            None,
            criterion,
            device,
            scaler,
            train_mode=False,
        )

        hist_train_loss.append(tr_loss)
        hist_val_loss.append(va_loss)
        hist_train_acc.append(tr_acc)
        hist_val_acc.append(va_acc)

        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss={tr_loss:.4f} train_seq_acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f} val_seq_acc={va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_seq_acc": va_acc,
                    "args": vars(args),
                },
                best_path,
            )

    xs = list(range(1, args.epochs + 1))
    plot_curve(xs, hist_train_loss, hist_val_loss, "CTC loss", log_dir / "loss.png")
    plot_curve(xs, hist_train_acc, hist_val_acc, "sequence accuracy", log_dir / "seq_acc.png")
    print(f"最佳验证序列准确率: {best_val_acc:.4f}，权重已保存: {best_path}")


if __name__ == "__main__":
    main()
