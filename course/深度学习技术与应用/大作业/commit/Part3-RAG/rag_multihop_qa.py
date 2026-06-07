#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 多跳问答 (Multi-hop QA) vLLM 原生并行加速版脚本
基于 vLLM 内置 Python API 直接进行高吞吐量的批量 (Batch) 推理。
自动支持 4张卡 (Tensor Parallelism = 4) 并行加速，配合 SentenceTransformers 批量编码实现极速多步问答。
不包含任何自动评测指标逻辑。内置多级短实体清洗器，且输出格式完全对齐你的 eval.py 标准评测数据格式。
"""

import os
import sys
import json
import re
import argparse
import pickle
import torch

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 直接从原生 vLLM 导入推理模块
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =================================─────────────
# 1. 默认路径与配置（完全对齐用户最新的物理路径与TP设置）
# =================================─────────────
USER = "zhouqilei"
DEFAULT_EMBEDDING_PATH = f"/mnt/workspace/{USER}/experiment_openPangu/experiment_openPangu1B_RAG/all-MiniLM-L6-v2"
DEFAULT_LLM_PATH = f"/opt/pangu/openPangu-Embedded-7B-V1.1/"
DEFAULT_CORPUS_PATH = f"/mnt/workspace/{USER}/HotpotQA/hotpot_fullwiki_corpus.json"

INDEX_SAVE_PATH = "./faiss_fullwiki_index.bin"
METADATA_SAVE_PATH = "./faiss_fullwiki_metadata.pkl"


# =================================─────────────
# 2. 向量检索库模块 (FAISS + SentenceTransformers)
# =================================─────────────
class VectorKnowledgeBase:
    def __init__(self, embedding_path, device="cpu"):
        print(f"[*] 正在从本地加载 Embedding 模型: {embedding_path}")
        self.embed_model = SentenceTransformer(embedding_path, device=device)
        self.dimension = self.embed_model.get_sentence_embedding_dimension()
        self.index = None
        self.corpus_data = []

    def build_index_from_json(self, json_path, batch_size=2048):
        """
        从 fullwiki 语料库构建 FAISS 索引，并持久化到本地
        自适应兼容 dict 和 list 格式的语料库数据结构
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"未找到语料库文件: {json_path}")
            
        print(f"[*] 开始读取语料库并解析: {json_path} ...")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_corpus = json.load(f)
        
        print("[*] 解析文本结构并进行预分批...")
        
        iterator = []
        if isinstance(raw_corpus, dict):
            iterator = raw_corpus.items()
        elif isinstance(raw_corpus, list):
            for item in raw_corpus:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    content = item.get("text", item.get("content", item.get("sentences", "")))
                    iterator.append((title, content))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    iterator.append((item[0], item[1]))
        else:
            raise TypeError(f"不支持的语料库 JSON 根格式类型: {type(raw_corpus)}")

        documents = []
        for title, content in iterator:
            if isinstance(content, list):
                text_content = " ".join(content)
            else:
                text_content = str(content)
            documents.append({
                "title": title,
                "text": text_content,
                "full_content": f"Title: {title}\nContent: {text_content}"
            })
        
        self.corpus_data = documents
        total_docs = len(documents)
        print(f"[*] 语料库总文档数: {total_docs}，Embedding 维度: {self.dimension}")
        
        self.index = faiss.IndexFlatIP(self.dimension)
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_texts = [d["full_content"] for d in batch_docs]
            batch_embeds = self.embed_model.encode(
                batch_texts, 
                batch_size=256, 
                show_progress_bar=False, 
                convert_to_numpy=True
            )
            faiss.normalize_L2(batch_embeds)
            self.index.add(batch_embeds)
            print(f"    - 已构建 [ {i + len(batch_docs)} / {total_docs} ] 条文档索引")
            
        faiss.write_index(self.index, INDEX_SAVE_PATH)
        with open(METADATA_SAVE_PATH, "wb") as f:
            pickle.dump(self.corpus_data, f)
        print(f"[+] FAISS 索引构建完成！已成功持久化至 {INDEX_SAVE_PATH}")

    def load_index(self):
        """
        加载本地已持久化的 FAISS 索引与元数据
        """
        if os.path.exists(INDEX_SAVE_PATH) and os.path.exists(METADATA_SAVE_PATH):
            print(f"[*] 正在从本地缓存加载 FAISS 索引...")
            self.index = faiss.read_index(INDEX_SAVE_PATH)
            with open(METADATA_SAVE_PATH, "rb") as f:
                self.corpus_data = pickle.load(f)
            print(f"[+] 加载成功！当前索引文档数: {len(self.corpus_data)}")
            return True
        return False

    def search_batch(self, queries: list[str], top_k=3):
        """
        批量语义检索：利用 FAISS 内部的多线程并发，一次性输出整批 Query 的 Top-K 段落
        """
        if self.index is None:
            raise ValueError("FAISS 索引未加载，请先 build_index_from_json 或 load_index")
            
        query_vectors = self.embed_model.encode(queries, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(query_vectors)
        
        distances, indices = self.index.search(query_vectors, top_k)
        
        batch_results = []
        for i in range(len(queries)):
            results = []
            for score, idx in zip(distances[i], indices[i]):
                if idx < 0 or idx >= len(self.corpus_data):
                    continue
                doc = self.corpus_data[idx]
                results.append({
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": float(score)
                })
            batch_results.append(results)
        return batch_results


# =================================─────────────
# 3. 最终答案提取与后处理（深度清洗废话，提取极简短实体）
# =================================─────────────
_PANGU_NOISE_TOKENS = re.compile(r"\[unused(?:1[0-6]|1?[0-9])\]|<think>|</think>|<answer>|</answer>|<s>|</s>")

# 答案引导词提取正则
_LEAD_IN_PATTERNS = [
    re.compile(r"^(therefore,?\s+)?the\s+answer\s+is\s+(?:clearly\s+stated\s+as\s+)?['\"“]?([^'\"”\.]+)", re.IGNORECASE),
    re.compile(r"^(final\s+)?answer:\s*['\"“]?([^'\"”\.]+)", re.IGNORECASE),
    re.compile(r"^so,?\s+the\s+correct\s+answer\s+is\s+['\"“]?([^'\"”\.]+)", re.IGNORECASE),
    re.compile(r"^so,?\s+the\s+most\s+accurate\s+answer\s+is\s+['\"“]?([^'\"”\.]+)", re.IGNORECASE),
    re.compile(r"^therefore,?\s+the\s+answer\s+is\s+['\"“]?([^'\"”\.]+)", re.IGNORECASE),
]

def clean_output(text: str) -> str:
    """
    清洗大模型输出文本（优先首行策略 + 多级兜底）：
    
    核心发现：openPangu 在 "Answer:" 后面吐出的第一行几乎总是最精简、最正确的答案实体，
    后续的行全是它的解释/反思/Wait/However 等废话。因此清洗策略应为：
    
    1. 移除特殊无用标记 [unused16] [unused17] 等。
    2. 按行拆分，优先取第一个非空且有意义的短行作为答案（首行优先策略）。
    3. 如果首行过长（>6 词），则尝试从中匹配引导词 "The answer is..." 提取实体。
    4. 如果首行以废话词开头（Wait/Now/Since），则向后搜索，取第一个短实体行。
    5. 最终兜底：去除所有前缀引导词，截取前6个词作为答案。
    """
    if not text:
        return ""
        
    # Step 1: 移除基础噪声和标签
    text = _PANGU_NOISE_TOKENS.sub("", text).strip()
    
    # Step 2: 按行拆分，找到有意义的行
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return text.strip(".,'\""" ")
    
    # Step 3: 首行优先策略
    first_line = lines[0]
    
    # 如果首行本身就很短（<=6词），大概率就是干净的实体答案，直接返回
    if len(first_line.split()) <= 6 and not re.match(r"^(wait|now|since|given|however|hmm|i think|let me|looking|re-reading|the question|but)\b", first_line, re.IGNORECASE):
        # 去除可能的前缀引导词
        cleaned = re.sub(r"^(the answer is|answer:|final answer:)\s*", "", first_line, flags=re.IGNORECASE)
        return cleaned.strip(".,'\""" -")
    
    # Step 4: 首行较长或以废话开头，尝试从所有行中匹配引导词模式
    for line in lines:
        for pattern in _LEAD_IN_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(2).strip(".,'\""" ")
    
    # Step 5: 遍历所有行，找到第一个足够短（<=6词）且非废话开头的行
    waste_prefixes = re.compile(r"^(wait|now|since|given|however|hmm|i think|let me|looking|re-reading|the question|but|i see|i need|the user|note|if you|-)\b", re.IGNORECASE)
    for line in lines:
        if len(line.split()) <= 6 and not waste_prefixes.match(line):
            cleaned = re.sub(r"^(the answer is|answer:|final answer:)\s*", "", line, flags=re.IGNORECASE)
            return cleaned.strip(".,'\""" -")
    
    # Step 6: 所有行都很长或都是废话。从首行尝试提取 "is [Entity]" 模式
    is_match = re.search(r"\bis\s+['\"]?([A-Z][a-zA-Z0-9\s',\-]+)", first_line)
    if is_match:
        entity = is_match.group(1).strip(".,'\""" ")
        # 截取不超过6词
        words = entity.split()
        return " ".join(words[:6]).strip(".,'\""" ")
    
    # Step 7: 终极兜底——去掉引导词，取首行前6个词
    cleaned = re.sub(r"^(the answer is|answer:|final answer:|therefore,?\s*the answer is)\s*", "", first_line, flags=re.IGNORECASE)
    words = cleaned.split()
    return " ".join(words[:6]).strip(".,'\""" -")


# =================================─────────────
# 4. 主执行流 (Pipeline)
# =================================─────────────
def parse_args():
    parser = argparse.ArgumentParser(description="原生 vLLM 并行加速多跳 RAG 推理")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_PATH, help="SentenceTransformers 路径")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_PATH, help="openPangu 模型物理路径")
    parser.add_argument("--corpus_json", type=str, default=DEFAULT_CORPUS_PATH, help="fullwiki 原始 json 库路径")
    
    # 批量推理输入与输出
    parser.add_argument("--input_json", type=str, required=True, help="批量推理输入的 json 文件路径（例如 hotpot_test_v1.json）")
    parser.add_argument("--output_json", type=str, default="predictions_prompt.json", help="批量预测输出结果保存路径")
    
    # vLLM 原生并行参数
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="GPU/NPU 并行张量切割大小（TP，使用4张卡就设为4）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="NPU 显存利用率比例")
    parser.add_argument("--max_model_len", type=int, default=4096, help="大模型支持的最大上下文序列长度")
    
    # 检索超参数
    parser.add_argument("--hops", type=int, default=2, help="多跳 RAG 的跳数 (Hops)")
    parser.add_argument("--top_k", type=int, default=5, help="每跳检索召回的文档数 (Top-K，适当增大以提升召回率)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Step 1: 加载/构建 向量知识库（检索端）
    kb = VectorKnowledgeBase(args.embedding_model, device="cpu")
    if not kb.load_index():
        print("[*] 本地未检测到已有的 FAISS 索引缓存，开始首次冷启动构建...")
        kb.build_index_from_json(args.corpus_json)

    # Step 2: 加载原生 vLLM 引擎（大模型端，TP 并行）
    print(f"[*] 正在初始化原生 vLLM 推理引擎 (TensorParallel={args.tensor_parallel_size}) ...")
    
    llm = LLM(
        model=args.llm_model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    
    # 增加 repetition_penalty，大幅减少或消灭 loops 现象
    sampling_params = SamplingParams(
        temperature=0.0, # 评测用 Greedy Decoding
        max_tokens=100,  # CoT 中间步骤可以稍长
        repetition_penalty=1.15, # 重复惩罚调优，杜绝 'individual' 退化现象
    )
    
    # 最终答案生成专用参数：极短输出 + 换行即停
    answer_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=30,   # 最终答案最多 30 token，绝对逼迫模型只输出核心实体
        repetition_penalty=1.15,
        stop=["\n", "Question:", "Example", "Documents:"],  # 一旦换行或出现新问题标志，立刻截断
    )
    
    # 读取批量问题
    print(f"[*] 正在从读取测试文件: {args.input_json}")
    with open(args.input_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    total = len(test_data)
    print(f"[+] 加载完成，共加载 {total} 条待预测样本。开始高吞吐多跳流水线...")

    # =================================─────────────
    # 5. 高性能批量多跳控制流水线 (Fully-Batched RAG)
    # =================================─────────────
    rag_states = {i: {"context_docs": [], "retrieved_titles": set()} for i in range(total)}
    current_queries = [item["question"] for item in test_data]
    

    # 日志文件保存路径
    log_path = "retrieval_analysis.log"
    log_lines = []

    for hop in range(args.hops):
        print(f"\n[*] ===== 开始第 {hop + 1} 跳检索与推理流水线 (Hops {hop + 1}/{args.hops}) =====")
        # 5.1 批量并发语义检索 (Batch Vector Search)
        print(f"    - [Step 5.1] 正在对 {total} 个 Query 并发进行 FAISS 向量检索...")
        batch_results = kb.search_batch(current_queries, top_k=args.top_k)

        # 检索分析日志：输出每条样本的检索query、召回文档title、gold_answer，并保存到log_lines
        log_lines.append(f"\n[分析日志] 第{hop+1}跳检索分析：")
        for i in range(total):
            log_lines.append(f"样本{i+1}:\n  Query: {current_queries[i]}\n  Gold Answer: {test_data[i].get('answer', test_data[i].get('gold_answer', ''))}")
            log_lines.append(f"  Retrieved Titles: {[doc['title'] for doc in batch_results[i]]}")
        # 同步打印到控制台
        print(f"\n[分析日志] 第{hop+1}跳检索分析：")
        for i in range(total):
            print(f"样本{i+1}:\n  Query: {current_queries[i]}\n  Gold Answer: {test_data[i].get('answer', test_data[i].get('gold_answer', ''))}")
            print(f"  Retrieved Titles: {[doc['title'] for doc in batch_results[i]]}")

        # 推理结束后保存日志到文件
        with open(log_path, "w", encoding="utf-8") as flog:
            flog.write("\n".join(log_lines))
        print(f"\n[+] 检索分析日志已保存至: {log_path}")

        # 5.2 过滤与去重更新上下文
        for i in range(total):
            state = rag_states[i]
            for doc in batch_results[i]:
                if doc["title"] not in state["retrieved_titles"]:
                    state["retrieved_titles"].add(doc["title"])
                    formatted_doc = f"Document <{doc['title']}>: {doc['text']}"
                    state["context_docs"].append(formatted_doc)

        # 5.3 如果不是最后一跳，批量并行生成下一跳的 CoT（思维链）作为下一步的 Query 修正
        if hop < args.hops - 1:
            print(f"    - [Step 5.2] 正在构建 CoT 生成提示词并提交 vLLM Batch 推理...")
            prompts = []
            for i in range(total):
                context_str = "\n".join(rag_states[i]["context_docs"])
                cot_prompt = (
                    f"Background Documents:\n{context_str}\n\n"
                    f"Question: {test_data[i]['question']}\n"
                    f"Reason step-by-step from the background to find key missing clues. "
                    f"What information do we need to search next? Provide a single clear thinking step:"
                )
                prompts.append(cot_prompt)

            outputs = llm.generate(prompts, sampling_params)

            # 提取 CoT，并更新下一跳的检索 Query
            next_queries = []
            for i, output in enumerate(outputs):
                raw_text = output.outputs[0].text if output.outputs else ""
                cot_step = clean_output(raw_text)
                next_queries.append(f"{test_data[i]['question']} {cot_step}")

            current_queries = next_queries

    # =================================─────────────
    # 6. 最终 Batch 生成最终答案（Few-shot 强制短实体格式）
    # =================================─────────────
    print(f"\n[*] ===== 开始批量最终答案预测生成 =====")
    final_prompts = []
    for i in range(total):
        final_context = "\n\n".join(rag_states[i]["context_docs"])
        final_prompt = (
            f"Answer the question using ONLY the background documents. "
            f"Output ONLY the answer entity (a name, date, number, or place). "
            f"Do NOT explain. Do NOT repeat the question.\n\n"
            f"Example 1:\n"
            f"Documents: Document <Christopher Nolan>: Christopher Edward Nolan is a British-American film director...\n"
            f"Question: Who directed Interstellar?\n"
            f"Answer: Christopher Nolan\n\n"
            f"Example 2:\n"
            f"Documents: Document <University of California, Berkeley>: Founded in 1868...\n"
            f"Question: When was UC Berkeley founded?\n"
            f"Answer: 1868\n\n"
            f"Now answer:\n"
            f"Documents:\n{final_context}\n\n"
            f"Question: {test_data[i]['question']}\n"
            f"Answer:"
        )
        final_prompts.append(final_prompt)

    # vLLM 进行极速的最后一轮答案预测生成（使用 answer 专用超参）
    final_outputs = llm.generate(final_prompts, answer_sampling_params)
    
    # 拼接保存格式（完全对齐 eval.py 的 List[dict] 结构，每个 item 带 id, question, gold_answer, prediction, type）
    predictions_json = []
    for i, output in enumerate(final_outputs):
        raw_text = output.outputs[0].text if output.outputs else ""
        pred_answer = clean_output(raw_text)
        item = test_data[i]
        
        predictions_json.append({
            "id": item.get("id", ""),
            "question": item["question"],
            "gold_answer": item.get("answer", ""),  # HotpotQA 的答案键值为 'answer'
            "prediction": pred_answer,
            "raw_output": raw_text,
            "type": item.get("type", ""),
            "retrieved_documents": list(rag_states[i]["retrieved_titles"])  # 附带保存检索命中记录
        })
        
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions_json, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*60)
    print(f"[+] 多卡 vLLM 原生并行多跳推理完成！")
    print(f"[+] 预测结果已保存至: {args.output_json}")
    print(f"[+] 该预测文件已完美适配你的 eval.py 评测数据格式！")
    print("="*60)


if __name__ == "__main__":
    main()
