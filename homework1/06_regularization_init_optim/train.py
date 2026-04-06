"""
实验 6：在「最优结构」784-256-128-10 上对比：
- 两种以上正则化：L2 权重衰减、Dropout、以及二者组合；
- 两种以上初始化：Xavier、He、小方差高斯；
- 两种以上学习率优化：带动量的 SGD、RMSprop、Adam。

单张 PNG 含三个子图，分别对应上述三类对比（测试集准确率曲线）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nn_numpy import (
    MLPNumpy,
    accuracy,
    load_mnist,
    train_epoch_sgd,
    build_optimizer_state,
)


def run_training(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    init_method: str,
    dropout_p: float,
    l2_lambda: float,
    optimizer: str,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int = 42,
) -> list[float]:
    model = MLPNumpy(
        input_dim=784,
        hidden_dims=[256, 128],
        num_classes=10,
        init_method=init_method,
        dropout_p=dropout_p,
        seed=seed,
    )
    opt_state = build_optimizer_state(model, optimizer)
    test_accs: list[float] = []
    for _ in range(epochs):
        train_epoch_sgd(
            model,
            x_train,
            y_train,
            batch_size=batch_size,
            lr=lr,
            l2_lambda=l2_lambda,
            optimizer=optimizer,
            opt_state=opt_state,
        )
        logits_te = model.forward(x_test, train=False)
        test_accs.append(accuracy(logits_te, y_test))
    return test_accs


def main() -> None:
    np.random.seed(42)
    x_train, y_train, x_test, y_test = load_mnist()
    epochs = 35
    batch_size = 128

    # --- 正则化对比（固定 He 初始化 + Adam）---
    reg_specs = [
        ("L2 (λ=1e-3)", "he", 0.0, 1e-3, "adam", 0.002),
        ("Dropout (p=0.2)", "he", 0.2, 0.0, "adam", 0.002),
        ("L2+Dropout", "he", 0.15, 5e-4, "adam", 0.002),
    ]
    curves_reg: list[tuple[str, list[float]]] = []
    for label, init_m, do, l2, opt, lr in reg_specs:
        acc = run_training(
            x_train,
            y_train,
            x_test,
            y_test,
            init_m,
            do,
            l2,
            opt,
            lr,
            epochs,
            batch_size,
            seed=42,
        )
        curves_reg.append((label, acc))

    # --- 初始化对比（无额外正则，SGD 固定学习率）---
    init_specs = [
        ("Xavier", "xavier", 0.0, 0.0, "sgd", 0.35),
        ("He", "he", 0.0, 0.0, "sgd", 0.35),
        ("小方差高斯", "normal_small", 0.0, 0.0, "sgd", 0.35),
    ]
    curves_init: list[tuple[str, list[float]]] = []
    for label, init_m, do, l2, opt, lr in init_specs:
        acc = run_training(
            x_train,
            y_train,
            x_test,
            y_test,
            init_m,
            do,
            l2,
            opt,
            lr,
            epochs,
            batch_size,
            seed=43,
        )
        curves_init.append((label, acc))

    # --- 学习率优化算法对比（He 初始化）---
    optim_specs = [
        ("Momentum SGD", "he", 0.0, 0.0, "momentum", 0.05),
        ("RMSprop", "he", 0.0, 0.0, "rmsprop", 0.001),
        ("Adam", "he", 0.0, 0.0, "adam", 0.002),
    ]
    curves_opt: list[tuple[str, list[float]]] = []
    for label, init_m, do, l2, opt, lr in optim_specs:
        acc = run_training(
            x_train,
            y_train,
            x_test,
            y_test,
            init_m,
            do,
            l2,
            opt,
            lr,
            epochs,
            batch_size,
            seed=44,
        )
        curves_opt.append((label, acc))

    out_dir = Path(__file__).resolve().parent
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    xs = np.arange(1, epochs + 1)

    for ax, curves, title in zip(
        axes,
        [curves_reg, curves_init, curves_opt],
        ["正则化方法（测试准确率）", "参数初始化（测试准确率）", "学习率优化算法（测试准确率）"],
    ):
        for label, acc in curves:
            ax.plot(xs, acc, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("正则化 / 初始化 / 优化器 对比（结构 784-256-128-10）")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=150)
    print(f"已保存 {out_dir / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
