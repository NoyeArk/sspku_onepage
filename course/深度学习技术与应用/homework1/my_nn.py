"""
纯 NumPy 多层感知机：前向传播、手动反向传播、多种训练与正则化选项。
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml


def softmax(z: np.ndarray) -> np.ndarray:
    # 这里为了防止指数爆炸，减去最大值
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def softmax_cross_entropy_loss(
    logits: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    n = logits.shape[0]
    probs = softmax(logits)
    log_likelihood = -np.log(probs[range(n), y] + 1e-12)

    # loss 计算均值
    loss = float(np.mean(log_likelihood))
    grad = probs.copy()
    grad[range(n), y] -= 1.0
    grad /= n
    return loss, grad


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_backward(z: np.ndarray, grad: np.ndarray) -> np.ndarray:
    return grad * (z > 0)


def dropout_forward(
    x: np.ndarray, p: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray | None]:
    if p <= 0:
        return x, None
    mask = (rng.random(x.shape) > p).astype(np.float64)
    scale = 1.0 / (1.0 - p)
    return x * mask * scale, mask


def dropout_backward(grad: np.ndarray, mask: np.ndarray | None, p: float) -> np.ndarray:
    if p <= 0 or mask is None:
        return grad
    scale = 1.0 / (1.0 - p)
    return grad * mask * scale


def init_weights(
    fan_in: int,
    fan_out: int,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    参数初始化

    Args:
        fan_in (`int`): 输入维度
        fan_out (`int`): 输出维度
        method (`str`): 'xavier', 'he', 'normal_small'
        rng (`np.random.Generator`): 随机数生成器

    Returns:
        W (`np.ndarray`): 权重矩阵
        b (`np.ndarray`): 偏置向量
    """
    if method == "xavier":
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W = rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)
    elif method == "he":
        std = np.sqrt(2.0 / fan_in)
        W = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
    else:
        W = rng.normal(0.0, 0.01, size=(fan_in, fan_out)).astype(np.float64)
    b = np.zeros(fan_out, dtype=np.float64)
    return W, b


class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        init_method: str = "he",
        dropout_p: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.dropout_p = dropout_p
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            W, b = init_weights(dims[i], dims[i + 1], init_method, self.rng)
            self.W.append(W)
            self.b.append(b)
        self._cache: dict = {}

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._cache = {"xs": [x], "zs": [], "masks": []}
        h = x
        L = len(self.W)

        # 前 L - 1 层：ReLU + Dropout
        for layer_idx in range(L - 1):
            z = h @ self.W[layer_idx] + self.b[layer_idx]
            self._cache["zs"].append(z)
            a = relu(z)
            if train and self.dropout_p > 0:
                a, mask = dropout_forward(a, self.dropout_p, self.rng)
                self._cache["masks"].append(mask)
            else:
                self._cache["masks"].append(None)
            self._cache["xs"].append(a)
            h = a

        # 最后一个线性层
        logits = h @ self.W[-1] + self.b[-1]
        self._cache["logits"] = logits
        return logits

    def backward(
        self, grad_logits: np.ndarray, l2_lambda: float = 0.0
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        手动反向传播，返回每层 (dW, db)。

        Args:
            grad_logits (`np.ndarray`): 损失对 logits 的梯度
            l2_lambda (`float`): L2 系数；0 表示不加 L2

        Returns:
            grads (`list`): 与 self.W 同序的梯度列表
        """
        grads_W: list[np.ndarray] = []
        grads_b: list[np.ndarray] = []
        L = len(self.W)
        g = grad_logits
        h_prev = self._cache["xs"][-1]
        dW = h_prev.T @ g
        if l2_lambda > 0:
            dW += l2_lambda * self.W[-1]
        db = np.sum(g, axis=0)
        grads_W.insert(0, dW)
        grads_b.insert(0, db)
        g = g @ self.W[-1].T

        for layer_idx in range(L - 2, -1, -1):
            mask = self._cache["masks"][layer_idx]
            g = dropout_backward(g, mask, self.dropout_p)
            z = self._cache["zs"][layer_idx]
            g = relu_backward(z, g)
            h_in = self._cache["xs"][layer_idx]
            dW = h_in.T @ g
            if l2_lambda > 0:
                dW += l2_lambda * self.W[layer_idx]
            db = np.sum(g, axis=0)
            grads_W.insert(0, dW)
            grads_b.insert(0, db)
            g = g @ self.W[layer_idx].T
        return list(zip(grads_W, grads_b))

    def apply_sgd(self, grads: list[tuple[np.ndarray, np.ndarray]], lr: float) -> None:
        for i, (dW, db) in enumerate(grads):
            self.W[i] -= lr * dW
            self.b[i] -= lr * db

    def apply_momentum(
        self,
        grads: list[tuple[np.ndarray, np.ndarray]],
        lr: float,
        momentum: float,
        velocity: list[list[np.ndarray]],
    ) -> None:
        for i, (dW, db) in enumerate(grads):
            velocity[i][0] = momentum * velocity[i][0] + dW
            velocity[i][1] = momentum * velocity[i][1] + db
            self.W[i] -= lr * velocity[i][0]
            self.b[i] -= lr * velocity[i][1]

    def apply_rmsprop(
        self,
        grads: list[tuple[np.ndarray, np.ndarray]],
        lr: float,
        cache: list[list[np.ndarray]],
        decay: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        for i, (dW, db) in enumerate(grads):
            cache[i][0] = decay * cache[i][0] + (1 - decay) * (dW**2)
            cache[i][1] = decay * cache[i][1] + (1 - decay) * (db**2)
            self.W[i] -= lr * dW / (np.sqrt(cache[i][0]) + eps)
            self.b[i] -= lr * db / (np.sqrt(cache[i][1]) + eps)

    def apply_adam(
        self,
        grads: list[tuple[np.ndarray, np.ndarray]],
        lr: float,
        m: list[list[np.ndarray]],
        v: list[list[np.ndarray]],
        t: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        for i, (dW, db) in enumerate(grads):
            m[i][0] = beta1 * m[i][0] + (1 - beta1) * dW
            m[i][1] = beta1 * m[i][1] + (1 - beta1) * db
            v[i][0] = beta2 * v[i][0] + (1 - beta2) * (dW**2)
            v[i][1] = beta2 * v[i][1] + (1 - beta2) * (db**2)
            m_hat_w = m[i][0] / (1 - beta1**t)
            m_hat_b = m[i][1] / (1 - beta1**t)
            v_hat_w = v[i][0] / (1 - beta2**t)
            v_hat_b = v[i][1] / (1 - beta2**t)
            self.W[i] -= lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
            self.b[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从 sklearn 加载 MNIST，像素归一化到 [0,1]。"""
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = mnist["data"].astype(np.float64) / 255.0
    y = mnist["target"].astype(np.int64)
    n_train = 60000
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]
    return x_train, y_train, x_test, y_test


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y))


def train_epoch_sgd(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    lr: float,
    l2_lambda: float,
    optimizer: str,
    opt_state: dict | None,
) -> tuple[float, float]:
    n = x.shape[0]
    total_loss = 0.0
    batches = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb, yb = x[start:end], y[start:end]

        # 前向传播
        logits = model.forward(xb, train=True)
        loss, g = softmax_cross_entropy_loss(logits, yb)
        if l2_lambda > 0:
            for W in model.W:
                loss += 0.5 * l2_lambda * float(np.sum(W**2)) / xb.shape[0]
        grads = model.backward(g, l2_lambda=l2_lambda)
        if optimizer == "sgd":
            model.apply_sgd(grads, lr)
        elif optimizer == "momentum":
            model.apply_momentum(
                grads, lr, opt_state["momentum"], opt_state["velocity"]
            )
        elif optimizer == "rmsprop":
            model.apply_rmsprop(grads, lr, opt_state["cache"])
        elif optimizer == "adam":
            opt_state["t"] += 1
            model.apply_adam(grads, lr, opt_state["m"], opt_state["v"], opt_state["t"])
        total_loss += loss * (end - start)
        batches += 1
    avg_loss = total_loss / n
    logits_full = model.forward(x, train=False)
    acc = accuracy(logits_full, y)
    return avg_loss, acc


def train_epoch_bgd(
    model: MLPNumpy,
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2_lambda: float,
) -> tuple[float, float]:
    """单 epoch：全批量梯度下降（BGD）。"""
    logits = model.forward(x, train=True)
    loss, g = softmax_cross_entropy_loss(logits, y)
    if l2_lambda > 0:
        for W in model.W:
            loss += 0.5 * l2_lambda * float(np.sum(W**2)) / x.shape[0]
    grads = model.backward(g, l2_lambda=l2_lambda)
    model.apply_sgd(grads, lr)
    logits_eval = model.forward(x, train=False)
    acc = accuracy(logits_eval, y)
    return loss, acc


def build_optimizer_state(model: MLP, optimizer: str) -> dict:
    L = len(model.W)
    if optimizer == "momentum":
        velocity = [
            [np.zeros_like(model.W[i]), np.zeros_like(model.b[i])] for i in range(L)
        ]
        return {"momentum": 0.9, "velocity": velocity}
    if optimizer == "rmsprop":
        cache = [
            [np.zeros_like(model.W[i]), np.zeros_like(model.b[i])] for i in range(L)
        ]
        return {"cache": cache}
    if optimizer == "adam":
        m = [[np.zeros_like(model.W[i]), np.zeros_like(model.b[i])] for i in range(L)]
        v = [[np.zeros_like(model.W[i]), np.zeros_like(model.b[i])] for i in range(L)]
        return {"m": m, "v": v, "t": 0}
    return {}
