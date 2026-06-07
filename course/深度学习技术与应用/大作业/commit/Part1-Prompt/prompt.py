#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# Part 1 — Prompting: HotpotQA 多跳问答推理 + 评测（vLLM 加速版）
# 基于 openPangu-Embedded-7B-V1.1，纯 Prompt 工程（不使用检索）
# =============================================================================

import json
import re
import os
import sys
import string
import time
import logging
from collections import Counter
from datetime import datetime

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 0. 配置区 —— 请根据实际情况修改以下路径和参数
# =============================================================================

USER = "zhouqilei"

# 路径配置
BASE_DIR = f"/mnt/workspace/{USER}/experiment_openPangu/experiment_openPangu1B_RAG"
MODEL_PATH = f"/mnt/workspace/{USER}/openPangu-Embedded-7B-V1.1/"
DATA_PATH = os.path.join(BASE_DIR, "hotpot_qa", "hotpot_test_v1.json")

# 输出路径
OUTPUT_DIR = f"/mnt/workspace/{USER}/experiment_openPangu/openPangu7B-Prompt/prompt_output"
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions_prompt.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "prompt_inference.log")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results_prompt.json")

# vLLM 引擎参数
TENSOR_PARALLEL_SIZE = 1         # 单卡=1；多卡时设置卡数
GPU_MEMORY_UTILIZATION = 0.90   # NPU 显存利用率
MAX_MODEL_LEN = 4096             # 最大序列长度（prompt + 生成）

# 生成参数
MAX_TOKENS = 256                 # /no_think 生效后实际输出很短，256 兜底
TEMPERATURE = 0.0                # 评测用 greedy decoding
TOP_P = 1.0
REPETITION_PENALTY = 1.05       # 轻微惩罚重复，防止循环输出

# 是否启用 thinking 模式
# False = 拼接 /no_think 关思考，直接出答案
USE_THINKING = False

# =============================================================================
# 1. 日志设置
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info(f"实验启动时间: {datetime.now()}")
logger.info(f"后端: vLLM (continuous batching)")
logger.info(f"用户名: {USER}")
logger.info(f"模型路径: {MODEL_PATH}")
logger.info(f"数据路径: {DATA_PATH}")
logger.info(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
logger.info(f"GPU Memory Utilization: {GPU_MEMORY_UTILIZATION}")
logger.info(f"Thinking 模式: {USE_THINKING}")
logger.info("=" * 60)


# =============================================================================
# 2. 加载 vLLM 引擎 & Tokenizer
# =============================================================================

def load_vllm_engine(model_path: str) -> tuple[LLM, AutoTokenizer]:
    """
    加载 vLLM 推理引擎和 tokenizer。

    vLLM 在初始化时自动完成：
    - 模型权重加载到 NPU
    - PagedAttention KV cache 预分配
    - continuous batching 调度器就绪

    openPangu 在 vLLM 下的写法与主流模型一致。
    """
    logger.info(f"正在加载 tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer 加载完成，vocab_size={tokenizer.vocab_size}")

    logger.info("正在初始化 vLLM 引擎（首次加载约需 30-60 秒）...")
    t0 = time.time()

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        # 以下为 NPU / 昇腾环境常用参数，按需开启
        # dtype="float16",
        # enforce_eager=True,       # 禁用 CUDA Graph，NPU 兼容性更好
        # max_num_batched_tokens=MAX_MODEL_LEN,  # 限制 batch 总 token 数
    )

    logger.info(f"vLLM 引擎初始化完成，耗时 {time.time()-t0:.1f}s")
    return llm, tokenizer


# =============================================================================
# 3. 数据加载
# =============================================================================

def load_hotpotqa_data(data_path: str) -> list[dict]:
    logger.info(f"正在加载数据: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"加载完成，共 {len(data)} 条数据")
    return data


# =============================================================================
# 4. Prompt 构建（批量）
# =============================================================================

def build_prompts(
    questions: list[str],
    tokenizer,
    use_thinking: bool = False,
) -> list[str]:
    """
    批量构建 Prompt —— 使用 tokenizer.apply_chat_template

    关键发现：
    - /no_think 放 system message → openPangu 不认，进入完整思考
    - /no_think 放 user_content 末尾 → 部分生效（思考内容清空但 token 残留）
    - /no_think 放 apply_chat_template 之后、prompt 末尾 → 最可靠

    经验证：/no_think 必须接在 chat template 生成的 prompt 字符串最末尾处。
    """
    prompts = []
    for question in questions:
        user_content = (
            f"Answer the following question concisely and directly.\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # /no_think 必须放在 apply_chat_template 之后、最终 prompt 末尾
        if not use_thinking:
            prompt += " /no_think"

        prompts.append(prompt)
    return prompts


# =============================================================================
# 5. 答案提取（增强版）
#
# 从实际运行结果看：
#   - /no_think 会抑制思考内容，但 [unused16][unused17] token 仍留在输出中
#   - 部分输出包含 <answer>...</answer> 标签
#   - 部分输出是直接文本
#
# 因此第一步就统一清理 [unused16] 和 [unused17]。
# =============================================================================

# Pangu 输出中的特殊 token（无论 thinking 开关都会出现）
_PANGU_NOISE_TOKENS = re.compile(
    r"\[unused(?:1[0-6]|1?[0-9])\]",  # [unused0] ~ [unused17]
)


def _strip_pangu_tokens(text: str) -> str:
    """移除 Pangu 输出中的所有 [unusedNN] token。"""
    return _PANGU_NOISE_TOKENS.sub("", text)


def _extract_answer_clean(text: str) -> str:
    """
    从清理后的文本中提取答案。

    策略（按优先级）：
    1. <answer>...</answer> 标签提取
    2. 取最后一段有意义的内容（多行时取最后一行）
    """
    text = text.strip()

    # 策略1：<answer>...</answer> 标签
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 策略2：取最后一段非空行（通常是答案）
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return text


def extract_answer_direct(raw_output: str) -> str:
    """
    从 /no_think 模式的输出中提取答案。

    实测中 Pangu 仍然输出 [unused16][unused17]，但中间无思考内容。
    所以先统一去除 token，再提取。
    """
    clean = _strip_pangu_tokens(raw_output)
    return _extract_answer_clean(clean)


def extract_answer_from_thinking(raw_output: str) -> str:
    """
    从 thinking 模式的输出中提取最终答案。

    输出格式：[unused16]思考过程...[unused17]最终答案
    注意 Pangu 不一定输出 [unused17]。

    策略：先整体清理 token，再提取答案。
    """
    clean = _strip_pangu_tokens(raw_output)

    # 如果清理后还有内容，按标准流程提取
    if clean.strip():
        return _extract_answer_clean(clean)

    # 兜底：尝试从原始输出中找最后的有意义内容
    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        last_clean = _strip_pangu_tokens(last)
        if last_clean.strip():
            return last_clean.strip()

    return raw_output.strip()


def extract_answers(
    raw_outputs: list[str],
    use_thinking: bool,
) -> list[str]:
    extract_fn = extract_answer_from_thinking if use_thinking else extract_answer_direct
    return [extract_fn(raw) for raw in raw_outputs]


# =============================================================================
# 6. vLLM 批量推理
# =============================================================================

def run_batch_inference(
    llm: LLM,
    prompts: list[str],
) -> list[str]:
    """
    使用 vLLM 对一批 prompt 进行推理，返回原始输出文本列表。

    vLLM 的 generate() 自动执行 continuous batching：
    - 内部动态打包请求，最大化 NPU 利用率
    - 比逐条 transformers.generate 快 10-50x
    """
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
    )

    outputs = llm.generate(prompts, sampling_params)

    # 提取生成的文本
    raw_outputs = []
    for output in outputs:
        if output.outputs:
            raw_outputs.append(output.outputs[0].text)
        else:
            raw_outputs.append("")  # 生成失败兜底

    return raw_outputs


# =============================================================================
# 7. 评测函数（文档统一标准，与 transformers 版完全一致）
# =============================================================================

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
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
    precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = num_same / len(gold_tokens) if len(gold_tokens) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def evaluate_answers(
    predictions: list[str],
    gold_answers: list[str],
    label: str = "",
) -> dict:
    total_em = sum(exact_match_score(p, g) for p, g in zip(predictions, gold_answers))
    total_f1 = sum(f1_score(p, g) for p, g in zip(predictions, gold_answers))
    n = len(predictions)
    avg_em = total_em / n if n > 0 else 0.0
    avg_f1 = total_f1 / n if n > 0 else 0.0

    result = {
        "label": label,
        "num_samples": n,
        "exact_match": round(avg_em * 100, 2),
        "f1": round(avg_f1 * 100, 2),
    }
    logger.info(f"[{label}] EM={avg_em*100:.2f}%  F1={avg_f1*100:.2f}%  (n={n})")
    return result


# =============================================================================
# 8. 主流程
# =============================================================================

def main():
    # 8.1 加载 vLLM 引擎
    llm, tokenizer = load_vllm_engine(MODEL_PATH)

    # 8.2 加载数据
    test_data = load_hotpotqa_data(DATA_PATH)

    # 可选：少量数据快速测试
    # test_data = test_data[:10]

    total = len(test_data)
    questions = [item["question"] for item in test_data]
    gold_answers = [item["answer"] for item in test_data]

    # 8.3 批量构建 Prompt
    logger.info(f"正在构建 {total} 条 Prompt ...")
    t0 = time.time()
    prompts = build_prompts(questions, tokenizer, use_thinking=USE_THINKING)
    logger.info(f"Prompt 构建完成，耗时 {time.time()-t0:.1f}s")

    # 8.4 vLLM 批量推理（一次性提交全部 200 条）
    logger.info(f"开始 vLLM 批量推理，共 {total} 条 ...")
    t0 = time.time()

    raw_outputs = run_batch_inference(llm, prompts)

    infer_time = time.time() - t0
    logger.info(f"vLLM 推理完成！")
    logger.info(f"  总耗时: {infer_time:.1f}s")
    logger.info(f"  吞吐量: {total/infer_time:.2f} 条/秒")
    logger.info(f"  平均每条约: {infer_time/total*1000:.0f} ms")

    # 8.5 答案提取
    logger.info("正在提取答案 ...")
    predictions = extract_answers(raw_outputs, USE_THINKING)

    # 打印前几条样例
    for i in range(min(5, total)):
        logger.info(f"\n--- 样例 {i+1} ---")
        logger.info(f"  Q: {questions[i][:80]}...")
        logger.info(f"  Gold: {gold_answers[i]}")
        logger.info(f"  Pred: {predictions[i]}")
        logger.info(f"  Raw : {raw_outputs[i][:200]}...")

    # 8.6 保存预测结果
    predictions_json = []
    for item, pred, raw in zip(test_data, predictions, raw_outputs):
        predictions_json.append({
            "id": item.get("id", ""),
            "question": item["question"],
            "gold_answer": item["answer"],
            "prediction": pred,
            "raw_output": raw,
            "type": item.get("type", ""),
        })

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions_json, f, ensure_ascii=False, indent=2)
    logger.info(f"\n预测结果已保存至: {PREDICTIONS_FILE}")

    # 8.7 评测
    logger.info("\n" + "=" * 60)
    logger.info("开始评测...")
    result = evaluate_answers(predictions, gold_answers, label="Part1-Prompt")

    # 分 type 评测
    logger.info("\n--- 分题型评测 ---")
    for qtype in ["bridge", "comparison"]:
        type_preds = [
            p for p, item in zip(predictions, test_data)
            if item.get("type") == qtype
        ]
        type_golds = [
            g for g, item in zip(gold_answers, test_data)
            if item.get("type") == qtype
        ]
        if type_preds:
            evaluate_answers(type_preds, type_golds, label=f"Part1-Prompt-{qtype}")

    # 8.8 保存评测结果
    results = {
        "overall": result,
        "config": {
            "backend": "vLLM",
            "user": USER,
            "model_path": MODEL_PATH,
            "data_path": DATA_PATH,
            "use_thinking": USE_THINKING,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "repetition_penalty": REPETITION_PENALTY,
            "num_samples": total,
            "inference_time_seconds": round(infer_time, 1),
            "throughput_items_per_sec": round(total / infer_time, 2),
        },
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"评测结果已保存至: {RESULTS_FILE}")
    logger.info("=" * 60)
    logger.info("实验完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()