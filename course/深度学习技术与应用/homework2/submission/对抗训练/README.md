## 目录结构

| 路径 | 说明 |
|------|------|
| `../../model/cnn_advtr.pt` | 对抗训练得到的新分类器权重 |
| `../../results/adversarial_train_meta.json` | 对抗训练过程与超参、test 准确率 |
| `../../attack_data/correct_1k_dual.pkl` | 评测用 1000 条：两模型在测试集上均预测正确的子集（与 `make_dual_correct_pkl.py` 一致） |
| `whitebox_baseline/` | 旧模型 `cnn_acc_90.53.pt` 上定向白盒 I-FGSM；`dual_1k_metrics.json` + `figures_dual/`（10 组原图/对抗图） |
| `whitebox_on_advtrained/` | **新模型** `cnn_advtr.pt` 上同一套白盒攻击；`dual_1k_metrics.json` + `figures_dual/`（作业要求的 10 组） |
| `01_blackbox_baseline_old_ckpt/` | 旧模型为受害者的黑盒协议（仅前向）；`metrics.json`、`figures/`、`predictions_table.csv` |
| `02_blackbox_on_advtrained/` | **新模型**为受害者的黑盒协议；`metrics.json`、`figures/`、`predictions_table.csv`（作业要求的 10 组） |