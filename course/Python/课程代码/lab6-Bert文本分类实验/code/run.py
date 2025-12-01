# coding: UTF-8
import time
import torch
import numpy as np
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(
    description="Chinese Text Classification and NER with BERT"
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    choices=["classification", "ner"],
    help="choose a task: classification or ner",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="choose a model: Bert, ERNIE, bert_RNN, bert_RCNN (all models support both classification and ner tasks)",
)
args = parser.parse_args()

if args.task == "classification":
    from train_eval import train, init_network
    from utils import build_dataset, build_iterator, get_time_dif

    # 默认数据集
    default_dataset = "toutiao-text-classfication-dataset"
elif args.task == "ner":
    from train_eval_ner import train
    from utils_ner import build_dataset, build_iterator, get_time_dif

    # 默认数据集
    default_dataset = "youku"
else:
    raise ValueError(f"Unknown task: {args.task}")

if __name__ == "__main__":
    dataset = default_dataset  # 数据集

    model_name = args.model  # 模型名称
    x = import_module("models." + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    # 启用cudnn benchmark以优化卷积操作（在确定性训练后启用）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
