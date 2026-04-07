"""
实验 6（子实验 1）：在结构 784-256-128-10 上对比正则化方法（L2、Dropout、组合）。
固定 He 初始化 + Adam。输出单张测试集准确率曲线 PNG。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from my_nn import (
    MLP,
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
    model = MLP(
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

    out_dir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(1, epochs + 1)
    for label, acc in curves_reg:
        ax.plot(xs, acc, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("正则化方法（测试准确率）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle("正则化对比（结构 784-256-128-10，He + Adam）")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve_regularization.png", dpi=150)
    print(f"已保存 {out_dir / 'learning_curve_regularization.png'}")


if __name__ == "__main__":
    main()
