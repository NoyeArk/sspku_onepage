"""
实验 6（子实验 3）：在结构 784-256-128-10 上对比优化算法（Momentum SGD、RMSprop、Adam）。
He 初始化，无额外正则。输出单张测试集准确率曲线 PNG。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from course.深度学习技术与应用.homework1.my_nn import (
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
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(1, epochs + 1)
    for label, acc in curves_opt:
        ax.plot(xs, acc, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("学习率优化算法（测试准确率）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle("优化器对比（结构 784-256-128-10，He 初始化）")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve_optimizers.png", dpi=150)
    print(f"已保存 {out_dir / 'learning_curve_optimizers.png'}")


if __name__ == "__main__":
    main()
