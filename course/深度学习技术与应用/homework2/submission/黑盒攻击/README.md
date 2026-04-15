# 第三章 C：黑盒攻击 — 提交物说明

## 实验结果摘要（1000 样本，`attack_data/correct_1k.pkl`）

| 实验 | 受害模型（仅前向） | 替代模型（I-FGSM 迁移） | 定向黑盒成功率 |
|------|-------------------|-------------------------|----------------|
| `01_provided_blackbox_model/` | `model/cnn.ckpt` | `cnn_acc_90.53.pt` | **8.3%**（83/1000） |
| `02_own_model_as_blackbox/` | `cnn_acc_90.53.pt` | `cnn.ckpt` | **1.3%**（13/1000） |

说明：`02` 为作业要求的「在自训模型上报告黑盒式成功率」，与后续对抗训练章节对比时使用；受害网络全程不求梯度。

## 目录

- `01_provided_blackbox_model/`：对**课程提供**的 `model/cnn.ckpt` 的定向黑盒攻击结果（作业主提交）。
- `02_own_model_as_blackbox/`：在**自训模型** `cnn_acc_90.53.pt` 上、仅使用前向接口的同一套攻击流程（供对抗训练章节对比）。

## 各子目录内容

- `metrics.json`：成功率、替代模型路径、随机种子、10 组可视化元数据。
- `predictions_table.csv`：10 组样本的受害模型在净图/对抗图上的预测类别。
- `figures/`：10 张原图 + 10 张对抗样本（灰度 PNG）。

## 方法说明

对待攻击模型**不反传**：仅在替代模型上做定向 I-FGSM 生成候选；若受害模型预测未达目标类，则在 L_inf 球内随机扰动并查询受害模型，直至查询上限。定向关系：原类 \(y\) → 目标类 \((y+1)\bmod 10\)。

## 复现实验

在 `Fashion-MNIST/code/` 下：

```bash
# 全流程（GPU 0）
python blackbox_attack.py --gpu 0 --submission_dir ../submission/chapter3_blackbox

# 若已跑完 01，仅补跑 02
python blackbox_attack.py --gpu 0 --only_own --submission_dir ../submission/chapter3_blackbox
```
