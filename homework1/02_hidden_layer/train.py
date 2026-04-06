"""
实验 2：在「最优配置」基础上增加一个隐藏层（结构变为 256-128-64）。
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


def main() -> None:
    np.random.seed(42)
    x_train, y_train, x_test, y_test = load_mnist()
    hidden_dims = [256, 128, 64]
    lr = 0.3
    batch_size = 128
    epochs = 45
    init_method = "normal_small"

    model = MLPNumpy(
        input_dim=784,
        hidden_dims=hidden_dims,
        num_classes=10,
        init_method=init_method,
        dropout_p=0.0,
        seed=42,
    )
    opt_state = build_optimizer_state(model, "sgd")

    train_losses: list[float] = []
    train_accs: list[float] = []
    test_accs: list[float] = []

    for ep in range(1, epochs + 1):
        tl, ta = train_epoch_sgd(
            model,
            x_train,
            y_train,
            batch_size=batch_size,
            lr=lr,
            l2_lambda=0.0,
            optimizer="sgd",
            opt_state=opt_state,
        )
        logits_te = model.forward(x_test, train=False)
        te_acc = accuracy(logits_te, y_test)
        train_losses.append(tl)
        train_accs.append(ta)
        test_accs.append(te_acc)
        if ep % 5 == 0 or ep == 1:
            print(
                f"epoch {ep:3d}  train_loss={tl:.4f}  train_acc={ta:.4f}  test_acc={te_acc:.4f}"
            )

    out_dir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, epochs + 1), train_losses, label="train loss", color="C0")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot(
        range(1, epochs + 1), train_accs, label="train acc", color="C1", linestyle="--"
    )
    ax2.plot(range(1, epochs + 1), test_accs, label="test acc", color="C2")
    ax2.set_ylabel("accuracy")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    ax.set_title("增加一个隐藏层（256-128-64）")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=150)
    print(f"已保存 {out_dir / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
