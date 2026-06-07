### openPangu实验指导

#### 0.背景知识

**1.1 开源项目概览**

openPangu-Embedded是华为诺亚方舟实验室自主研发的一系列小尺寸的开源大语言模型。2025年8月华为已将Pangu Embedded系列模型以**openPangu-Embedded**的名义开源，目前主要包括两个版本：

| 模型版本                   | 参数量                  | 训练数据量  | 定位                     |
| :------------------------- | :---------------------- | :---------- | :----------------------- |
| openPangu-Embedded-7B-V1.1 | 7B（不含词表Embedding） | ~25T tokens | 旗舰版，支持完整快慢思考 |
| openPangu-Embedded-1B-V1.1 | 1B                      | 未明确      | 轻量版，仅支持快思考模式 |

**1.2 开源代码**

openPangu开源项目托管在**GitCode平台**（华为及国内开源社区常用的代码托管服务），仓库地址为：

```
https://gitcode.com/ascend-tribe/openPangu-Embedded-7B-V1.1
```

**核心代码结构**：

- `inference/`：模型推理核心实现
- `modeling_openpangu_dense.py`：模型架构定义
- `configuration_openpangu_dense.py`：配置参数管理
- `inference/vllm_ascend/`：vLLM推理引擎的昇腾适配层（含注意力机制、量化模块等）

**1.3 相关论文**

除模型权重外，华为还发布了配套的技术论文：

《Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition》（arXiv:2505.22375），

《Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs》(arXiv: 2504.07866)



**核心技术点**

**2.1 双系统快慢思维架构**

**阶段一：基础推理机构建**

- 采用**模型感知的迭代蒸馏**：动态根据学生模型当前能力选择中等复杂度数据，避免数据过易或过难
- **迭代间模型合并**：通过参数增量累积，缓解灾难性遗忘
- **重复自修复机制**：局部n-gram检测+控制提示注入，打破生成循环
- **大规模强化学习**：基于GRPO算法，配合**多源自适应奖励系统（MARS）**，融合规则验证、LLM评估、偏好模型等多重奖励信号

**阶段二：双系统快慢思维**

- **手动切换**：用户通过元提示指定System 1（快速直接回答）或System 2（详细CoT推理）
- **自适应切换**：模型自动评估问题复杂度，对简单问题输出简洁答案（平均节省88% token），对复杂问题启用深度推理

| 模式                   | 触发方式              | 行为特征                                                   | 适用场景                     |
| :--------------------- | :-------------------- | :--------------------------------------------------------- | :--------------------------- |
| **慢思考（System 2）** | 默认模式              | 生成`<think>...</think>`标签包裹的详细推理过程，再输出答案 | 复杂数学、逻辑推理、代码生成 |
| **快思考（System 1）** | 添加`/no_think`标记   | 直接输出简洁答案，跳过中间推理                             | 简单问答、日常对话           |
| **自适应模式**         | 添加`/auto_think`标记 | 模型自动判断问题复杂度，动态选择思维模式                   | 混合难度场景                 |





#### 1.实验平台操作指导

使用IAM ID申请得到资源之后，登录[https://science.lab.huaweicloud.com/](https://science.lab.huaweicloud.com/lab/labResource)，左侧点击实验环境-开发环境（容器）。按以下步骤操作：

1）选择开发环境（容器），选择指定的环境；

2）点击启动，选择第一个euler2.9-py310-torch2.5.1-cann8.1.rc1-openPangu-Courses-notebook镜像；

3）启动完成后，选择 "去使用"；

![img](./images\env.png)

![img](./images\select.png)

启动中：

![img](./images\launching.png)

![img](./images\start.png)

工作目录/对应实际的目录/mnt/workspace，在当前页面下只能看到这个目录，一切页面相关的操作都在这里执行；

![img](./images\terminal.png)

模型所在目录：/opt/pangu/openPangu-Embedded-7B-V1.1

环境不设置隔离，请按照自己分配的环境进行实验，不要影响其他同学的实验；



#### 2.实验环境准备

**⭐实验中需要用用户名区分目录，为了规范，建议每个同学使用自己的完整姓名拼音；本文档里仅使用labs作为示例，请把labs替换成自己的用户名(如: User)**

由于容器中只有/opt/huawei/edu-apaas/src/init是持久化的，为了防止重启后文件丢失，需要在终端里执行如下命令，它会把/mnt/workspace/labs挂载到持久化的目录下；同时我们把必须的文件拷贝到这个目录下面；

```Shell
# 拷贝目录到/mnt/workspace/labs
USER=labs
mkdir -p /opt/huawei/edu-apaas/src/init/${USER}
ln -s /opt/huawei/edu-apaas/src/init/${USER} /mnt/workspace/${USER}
cp /opt/experiment_openPangu/ /mnt/workspace/${USER} -rf

MODEL_DIR=/opt/pangu/openPangu-Embedded-7B-V1.1
cp $MODEL_DIR -rf /mnt/workspace/labs/openPangu-Embedded-7B-V1.1
```

左边会出现目录：

![image-20260518201408689](.\images\image1.png)

如果是暂停后重启，会发现labs消失了，此时文件还在，只是软链接被删除了，

![img](./images\fig2.png)

要重新链接：

~~~
USER=labs
ln -s /opt/huawei/edu-apaas/src/init /mnt/workspace/${USER}
~~~

**不同容器之间的持久化目录是独立的；但是同一个容器内不同用户使用，要用目录区分**

注意事项：**请勿直接在** **`/opt/experiment_openPangu`** **目录下修改任何文件，以保证原始环境的完整可复用。如果担心造成后面的环境不可用，可以先备份到持久化目录下。**



#### 3.实验数据准备

HotpotQA测试数据集有两种设置如下

| Split Name      | # Examples | Usage           |
| --------------- | ---------- | --------------- |
| test-distractor | 7,405      | distractor test |
| test-fullwiki   | 7,405      | full wiki test  |

本实验中，我们选择fullwiki的数据；

为了统一测评标准，且减少测试所用的时间，我们从公开的fullwiki测试集采样200条数据，得到一个**统一的验证集** hotpot_test_v1.json，如下：

type=bridge: sampled 160 / 5918

type=comparison: sampled 40 / 1487

**网盘地址：**

https://disk.pku.edu.cn/link/AA60C61D354075487CA4296EE72FA53350 提取码semS

|                            | 下载地址/文件                | 备注                                   |
| -------------------------- | ---------------------------- | -------------------------------------- |
| 测试集                     | hotpot_test_v1.json          | 共200条                                |
| 训练集                     | train-00000-of-00001.parquet | 保存位置：${BASE_DIR}/hotpot_qa/       |
| RAG配套检索库test-fullwiki | hotpot_fullwiki_corpus.json  | 同上，可以自行选择其他更大规模的检索库 |

其中BASE_DIR=/mnt/workspace/labs/experiment_openPangu/experiment_openPangu1B_RAG/

HotpotQA Benchmark的参考：https://www.emergentmind.com/topics/hotpotqa-benchmark

#### 4.四类方法实现指导

##### 4.1. Prompt

本实验进行基于Prompt的评测，不使用检索、纯靠大模型提示工程完成 HotpotQA 多跳问答，作为后续其他方法的对比基准。

做这个实验之前先了解openPangu的基本输入输出方式，vllm，HotpotQA格式；

**1）openPangu的Special Token**

Pangu的special token跟Qwen模型的对比：

 [unused16] ~ <think>

 [unused17] ~ </think>

 但是实测使用中Pangu不一定会输出[unused17] ，从而影响解析，请自行处理解决。

 由于Pangu模型默认是thinking模式，所以如果要关闭thinking模式，要在prompt末尾 + " /no_think"。

 其他的token对应如下：

```Python
class PanguSftTemplate:
    system_token = "系统："
    user_token = "用户："
    assistant_token = "助手："
    tool_token = "工具："
    start_token = "[unused9]"
    end_token = "[unused10]"
```

Prompt格式：

~~~json
[
    {"role": "system", "content": ""},
    {"role": "user", "content": (
        f"Answer the following question concisely.\n"
        f"Question: {question}\n"
        f"Example: <answer>Paris</answer>\n"
        f"Answer:"
    )}
]
~~~

拼接prompt最好使用tokenizer的**apply_chat_template方法**，否则可能影响实验效果。

**2）训练集输入数据格式**

```JSON
[
  {
    "id": "5adf6a5c5542995ec70e8ff9",
    "question": "In what city does the most successful American and international five-and-dime business have a historic building?",
    "answer": "Watertown, New York",
    "type": "bridge"
  },
  ...
]
```

**3）评测函数(统一如下)：**

```Python
#!/usr/bin/env python
# coding: utf-8
# ─────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────
def normalize_answer(s):
    """Lowercase, remove articles/punctuation, collapse whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth):
    """Token-level F1 between normalized prediction and gold answer."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)
    if norm_pred in ['yes', 'no', 'noanswer'] and norm_pred != norm_gold:
        return 0.0
    if norm_gold in ['yes', 'no', 'noanswer'] and norm_pred != norm_gold:
        return 0.0
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    """1.0 if normalized prediction exactly matches gold, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def evaluate_answers(predictions, gold_data, label="", raw_outputs=None):
    """Compute and print average EM and F1."""
    # TODO: Implement evaluation logic
    pass
```

**4）模型加载/推理**

在已经安装了torch-npu等依赖库（容器已安装）的情况下，openPangu的vllm和transformers的写法跟主流的写法是相同的，所以请参考huggingface等代码的实现。

其他要求参考实验介绍文档



##### 4.2. SFT

**前置知识：**

**🔹NPU**

神经网络处理器。它是专门为加速神经网络计算而设计的硬件芯片，是昇腾AI生态的物理算力基础。

**🔹 mcore 格式**

​    mcore 格式是 MindSpeed-LLM 框架的内部权重标准，专门面向大规模分布式训练与加速器优化场景设计。

​    其核心特征包括：

​    \- 分布式分片存储：参数会根据 张量并行（Tensor Parallel） 与 流水线并行（Pipeline Parallel） 策略自动切分，每个 rank 仅加载自己负责的部分权重，显著降低显存占用；

​    \- 高效加载：mcore 权重在加载时无需重新聚合或拆分，可直接映射至设备内存，加速训练启动；

​    \- 硬件优化兼容：在 Ascend NPU 或其他加速平台下，mcore 格式权重匹配底层算子精度与存储布局（如权重排列、padding 对齐、半精度存储等），以实现更高计算效率。

**🔹 hf格式：**

Hugging Face平台定义的通用模型存储标准。它是目前开源社区最常见、即插即用的模型格式。

**🔹Megatron、MindSpeed、MindSpeed-LLM**

Megatron是英伟达推出的深度学习训练框架。它主要用于解决大模型训练中的显存瓶颈，率先实现了模型并行（张量、流水线并行），是大模型分布式训练领域的标杆框架，定义了后续许多工具的基础范式，而 MindSpeed 是华为昇腾为了让 Megatron 在昇腾 NPU 上高效运行而开发的加速适配插件。

MindSpeed-LLM是华为昇腾生态下的大模型训练框架。它构建于MindSpeed之上，在逻辑上复用并适配了Megatron的mcore架构，专门用于在昇腾芯片上高效执行大模型的分布式训练和微调。

![img](.\images\mindspeed.png)



**参考链接：**

**MindSpeed Core** https://gitcode.com/Ascend/MindSpeed

**MindSpeed-LLM** https://gitee.com/ascend/MindSpeed-LLM/tree/1.0.0/

**基于MindSpeed-LLM的openPangu-1B微调** https://github.com/minihash-999/openPangu-Embedded-1B-Finetune-demo



**实验要求：**

利用HotpotQA的训练集对openPangu-Embedded-7B模型进行SFT，并测试其在测试集上的表现。

**要点：**

SFT实验总体流程：训练数据格式转换、模型checkpoint hf格式转mcore格式、微调、训练checkpoint的mcore格式转hf格式、测试；请按照下文的指导进行；

**4.2.1 数据预处理**

实验环境中的MindSpeed-LLM是可以正常运行的，已经在容器中存在，其目录如下，我们设置为MINDSPEED_DIR常量：

~~~shell
MINDSPEED_DIR=/mnt/workspace/labs/experiment_openPangu/openPangu1B_SFT_INFER/experiment_openPangu1B_SFT/openPangu-Embedded-1B-Finetune-demo/MindSpeed-LLM
~~~



在训练之前，我们要把数据格式转换为用于SFT训练的格式；脚本为convert_hotpotqa_to_sft.py，见网盘；

~~~shell
cd $MINDSPEED_DIR
python convert_hotpotqa_to_sft.py
~~~

需要注意其内部参数改成自己的参数；



![image](./images\fig1.png)

生成文件在output_dir下，应该看到这样的结果：



![image](.\images\output.png)

**4.2.2 模型格式转换**

把openPangu-Embedded-7B-V1.1转成mcore格式以进行Megatron训练；

执行命令

```Shell
cd $MINDSPEED_DIR
python convert_ckpt.py --model-type GPT --load-model-type hf \
    --save-model-type mg --save-dir /mnt/workspace/labs/experiment_openPangu/openPangu1B_SFT_INFER/sft_outputs/ckpt/output/openPangu_7B_1_1_mcore \
    --load-dir /opt/pangu/openPangu-Embedded-7B-V1.1 \
    --tokenizer-model /opt/pangu/openPangu-Embedded-7B-V1.1 \
    --add-qkv-bias --add-dense-bias --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 1 --params-dtype bf16 --use-mcore-models
```

执行完毕会生成Megatron格式的checkpoint。

在后面训练完之后进行mcore到hf的转换；

**4.2.3 训练**

实验提供tune_pangu_7b_full_ptd.sh的脚本（北大网盘链接），其中关键参数部分请自己研究修改；

![image-20260518222744807](.\images\tune.png)

~~~shell
# 执行此命令！
cd $MINDSPEED_DIR && bash tune_pangu_7b_full_ptd.sh
~~~

1. 此外要根据前面的模型mcore输出目录修改路径，避免加载失败；训练所花的时间不会太长（<1H）；
2. 查看结果（注意改成自己生成的loss_log位置）

**4.2.4 mcore格式转换为hf**

进行格式转换；(参考4.2.2)

~~~shell
#!/bin/bash
MINDSPEED_DIR=/mnt/workspace/labs/experiment_openPangu/openPangu1B_SFT_INFER/experiment_openPangu1B_SFT/openPangu-Embedded-1B-Finetune-demo/MindSpeed-LLM

load_mcore_dir=/YOUR_PATH # ⭐ 修改
save_hf_dir=/opt/pangu/labs/openPangu-Embedded-7B-V1.1 # 必须是openPangu模型目录

export CUDA_DEVICE_MAX_CONNECTIONS=1

cd ${MINDSPEED_DIR}

echo "开始转换 mcore -> HF..."
echo "加载路径: ${load_mcore_dir}"
echo "保存路径: ${save_hf_dir}"

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --save-dir ${save_hf_dir} \
    --load-dir ${load_mcore_dir} \
    --add-qkv-bias \
    --add-dense-bias \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --params-dtype bf16 \
    --use-mcore-models

cp ${save_hf_dir}/tokenizer.model ${save_hf_dir}/mg2hf/
cp ${save_hf_dir}/tokenizer_config.json ${save_hf_dir}/mg2hf/
cp ${save_hf_dir}/tokenization_openpangu.py ${save_hf_dir}/mg2hf/
~~~



**4.2.5 推理/评测**

推理部分请自行完成；评测的指标参见上文的EM和F1；

如果要指定只使用1个GPU，则设置环境变量：

~~~
CUDA_VISIBLE_DEVICES=0 python inference.py
~~~



![image-20260518235416855](./images\inference.png)

另外，可以pip install ascend-nputop，用nputop命令查看GPU趋势。

![image](./images\npu-top.png)

##### 4.3. 基于RAG的多跳问答

基于**检索增强生成（RAG）** 实现**多跳问答（Multi-hop QA）**，使用fullwiki作为外部知识库，在 HotpotQA 数据集上完成问答推理与效果评估。

要求：

1. 掌握 RAG 系统的完整架构：**向量库构建 → 语义检索 →  大模型生成 → 答案评估**
2. 实现面向**多跳问题**的检索式问答（需融合多篇文档信息推理答案）

向量库构建可以参考Chroma，Faiss，Jina等，已经有很多成熟方案；

Embedding模型：/mnt/workspace/labs/experiment_openPangu/experiment_openPangu1B_RAG/all-MiniLM-L6-v2（也可用其他embedding模型，不做限制）。

检索数据库：章节3的hotpot_fullwiki_corpus.json。



##### 4.4. 基于Agent的多跳问答

实验要求使用Agent的方式进行多跳问答，Agent使用方式不限。具体实现不提供代码，请同学们自行调研完成。

参考实现：

基于 ReAct（Reasoning+Acting）范式的智能问答 Agent，集成大模型推理、向量库检索、多步迭代思考能力，核心是让大模型通过「思考 - 行动 - 观察」循环自主调用检索工具获取知识，最终生成答案。

可以利用 Serper 等检索 API 获取开放域下的网页信息。

![image](.\images\serper.png)

