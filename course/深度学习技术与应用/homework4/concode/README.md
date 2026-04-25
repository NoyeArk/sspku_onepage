# Mapping Language to Code in Programmatic Context

### 安装依赖

PyTorch 0.3（如使用 0.4 版本需做少量修改）
```
pip install antlr4-python3-runtime==4.6
pip install allennlp==0.3.0
pip install ipython
```

### 从 Google Drive 下载数据
```
mkdir concode
cd concode
```
从以下链接下载数据到该文件夹：https://drive.google.com/drive/folders/1kC6fe7JgOmEHhVFaXjzOmKeatTJy1I1W

### 生成产生式规则（Production Rules）

这将训练集限制为 **100,000** 条，验证/测试集各 **2,000** 条。如果你的计算资源允许，可以使用更多数据。

```bash
uv run build.py -train_file concode/train_shuffled_with_path_and_id_concode.json \
                -valid_file concode/valid_shuffled_with_path_and_id_concode.json \
                -test_file raw_concodedata/test_shuffled_with_path_and_id_concode.json \
                -output_folder data -train_num 100000 -valid_num 2000
```

```bash
uv run build.py -train_file raw_data/train.jsonl \
                -valid_file raw_data/dev.jsonl \
                -test_file raw_data/test.jsonl \
                -output_folder data -train_num 100000 -valid_num 2000
```

### 准备 PyTorch 数据集
```
mkdir data/d_100k_762
python preprocess.py -train data/train.dataset -valid data/valid.dataset \
                     -save_data data/d_100k_762/concode -train_max 100000 -valid_max 2000
```

### 训练模型

**Seq2Seq 模型：**
```bash
python train.py -dropout 0.5 -data data/d_100k_762/concode -save_model data/d_100k_762/s2s \
                -epochs 30 -learning_rate 0.001 -seed 1123 -enc_layers 2 -dec_layers 2 \
                -batch_size 50 -src_word_vec_size 1024 -tgt_word_vec_size 512 -rnn_size 1024 \
                -encoder_type regular -decoder_type regular -copy_attn
```

**Seq2Prod 模型：**
```bash
python train.py -dropout 0.5 -data data/d_100k_762/concode -save_model data/d_100k_762/s2p \
                -epochs 30 -learning_rate 0.001 -seed 1123 -enc_layers 2 -dec_layers 2 \
                -batch_size 20 -src_word_vec_size 1024 -tgt_word_vec_size 512 -rnn_size 1024 \
                -encoder_type regular -decoder_type prod -brnn -copy_attn
```

**Concode 模型（推荐）：**
```bash
uv run train.py -dropout 0.5 -data data/ -save_model data/ \
                -epochs 30 -learning_rate 0.001 -seed 1123 -enc_layers 2 -dec_layers 2 \
                -batch_size 20 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 \
                -decoder_rnn_size 1024 -encoder_type concode -decoder_type concode \
                -brnn -copy_attn -twostep -method_names -var_names
```

### 预测

**在验证集上：**
```
ipython predict.ipy -- -start 5 -end 30 -beam 3 -models_dir data/d_100k_762/concode/ \
                       -test_file data/valid.dataset -tgt_len 500
```

**在测试集上（使用验证集上的最佳 epoch）：**
```
ipython predict.ipy -- -start 15 -end 15 -beam 3 -models_dir data/d_100k_762/concode/ \
                       -test_file data/test.dataset -tgt_len 500
```

> 对于其他模型类型，请使用对应的 `-models_dir` 路径。