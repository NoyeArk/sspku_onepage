#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 4 — 基于 ReAct Agent 的多跳问答 (Multi-hop QA) —— 并行推理版
使用 openPangu-Embedded-7B-V1.1 + vLLM (4-NPU TP) + Serper API (Google 网页搜索)
实现 ReAct (Reasoning + Acting) 范式的智能问答 Agent

并行化策略：
- 每一步（step）中，所有尚未完成的样本的 prompt 被打包成一个 batch 一次性提交 vLLM
- 检索操作通过 ThreadPoolExecutor 多线程并发执行
- 已调用 Finish[] 的样本被标记为完成，不再参与后续步骤
- 极大程度发挥 vLLM Continuous Batching 的吞吐优势

参考论文：ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
"""

import os
import sys
import json
import re
import argparse
import pickle
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =================================─────────────
# 1. 配置区
# =================================─────────────
USER = "zhouqilei"
DEFAULT_EMBEDDING_PATH = f"/mnt/workspace/{USER}/experiment_openPangu/experiment_openPangu1B_RAG/all-MiniLM-L6-v2"
DEFAULT_LLM_PATH = "/opt/pangu/openPangu-Embedded-7B-V1.1/"
DEFAULT_CORPUS_PATH = f"/mnt/workspace/{USER}/HotpotQA/hotpot_fullwiki_corpus.json"

INDEX_SAVE_PATH = "/mnt/workspace/zhouqilei/experiment_openPangu/experiment_openPangu1B_RAG/faiss_fullwiki_index.bin"
METADATA_SAVE_PATH = "/mnt/workspace/zhouqilei/experiment_openPangu/experiment_openPangu1B_RAG/faiss_fullwiki_metadata.pkl"

# Serper API 配置
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "YOUR_SERPER_API_KEY_HERE")
SERPER_API_URL = "https://google.serper.dev/search"

# Agent 超参数
MAX_STEPS = 8
SEARCH_TOP_K = 5


# =================================─────────────
# 2. Serper API 网页搜索工具
# =================================─────────────
def serper_search(query: str, num_results: int = 3) -> str:
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}
    try:
        response = requests.post(SERPER_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            kg_text = f"[Knowledge Graph] {kg.get('title', '')}: {kg.get('description', '')}"
            if kg.get("attributes"):
                attrs = "; ".join([f"{k}: {v}" for k, v in kg["attributes"].items()])
                kg_text += f" | {attrs}"
            results.append(kg_text)
        if "answerBox" in data:
            ab = data["answerBox"]
            answer_text = ab.get("answer", ab.get("snippet", ""))
            if answer_text:
                results.insert(0, f"[Answer Box]: {answer_text}")
        for item in data.get("organic", [])[:num_results]:
            results.append(f"[{item.get('title', '')}]: {item.get('snippet', '')}")
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search API error: {str(e)}"


def is_serper_available() -> bool:
    return SERPER_API_KEY and SERPER_API_KEY != "YOUR_SERPER_API_KEY_HERE"


# =================================─────────────
# 3. 本地 FAISS 向量库（兜底检索）
# =================================─────────────
class VectorKnowledgeBase:
    def __init__(self, embedding_path, device="cpu"):
        print(f"[*] 加载 Embedding 模型: {embedding_path}")
        self.embed_model = SentenceTransformer(embedding_path, device=device)
        self.dimension = self.embed_model.get_sentence_embedding_dimension()
        self.index = None
        self.corpus_data = []

    def build_index_from_json(self, json_path, batch_size=2048):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"未找到语料库: {json_path}")
        print(f"[*] 读取语料库: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_corpus = json.load(f)
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
        documents = []
        for title, content in iterator:
            text_content = " ".join(content) if isinstance(content, list) else str(content)
            documents.append({"title": title, "text": text_content, "full_content": f"Title: {title}\nContent: {text_content}"})
        self.corpus_data = documents
        total_docs = len(documents)
        print(f"[*] 文档数: {total_docs}")
        self.index = faiss.IndexFlatIP(self.dimension)
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            embeds = self.embed_model.encode([d["full_content"] for d in batch], batch_size=256, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(embeds)
            self.index.add(embeds)
        faiss.write_index(self.index, INDEX_SAVE_PATH)
        with open(METADATA_SAVE_PATH, "wb") as f:
            pickle.dump(self.corpus_data, f)
        print("[+] FAISS 索引构建完成")

    def load_index(self):
        if os.path.exists(INDEX_SAVE_PATH) and os.path.exists(METADATA_SAVE_PATH):
            self.index = faiss.read_index(INDEX_SAVE_PATH)
            with open(METADATA_SAVE_PATH, "rb") as f:
                self.corpus_data = pickle.load(f)
            print(f"[+] FAISS 缓存加载成功，文档数: {len(self.corpus_data)}")
            return True
        return False

    def search(self, query, top_k=5):
        if self.index is None:
            return "No local index available."
        qv = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(qv)
        distances, indices = self.index.search(qv, top_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.corpus_data):
                doc = self.corpus_data[idx]
                results.append(f"[{doc['title']}]: {doc['text'][:300]}")
        return "\n".join(results) if results else "No relevant documents found."


# =================================─────────────
# 4. ReAct Agent Prompt 模板
# =================================─────────────
REACT_PROMPT_TEMPLATE = """You are a multi-hop question answering agent. You answer questions by iteratively searching for information and reasoning about it.

You have access to the following tools:
- Search[query]: Searches the knowledge base for relevant Wikipedia documents about the query.
- WebSearch[query]: Searches the web (Google) for additional information when the knowledge base is insufficient.
- Finish[answer]: Outputs your final answer. The answer must be a short entity (name, date, place, number, yes/no).

You MUST follow this format EXACTLY:

Thought: <your reasoning about what information you still need>
Action: Search[<specific search query>]

After receiving an Observation, continue with another Thought/Action cycle.
If the knowledge base results are insufficient, use WebSearch to find information from the web.

When you have enough information, use:
Thought: <your final reasoning combining all clues>
Action: Finish[<short answer>]

IMPORTANT RULES:
- Your answer in Finish[] must be ONLY the entity (1-5 words max). NO explanations.
- Always try Search (knowledge base) FIRST before using WebSearch.
- Each Search/WebSearch query should target ONE specific fact.
- You have at most {max_steps} steps total.
- For comparison questions (which is bigger/older/first), search for BOTH entities then compare.

Question: {question}
{scratchpad}"""


# =================================─────────────
# 5. 并行检索执行器
# =================================─────────────
def batch_search(queries_with_idx, use_serper, kb, max_workers=16):
    """
    对一批 (index, query, tool_type) 对进行并行检索
    tool_type: "local" 用本地 FAISS, "web" 用 Serper API
    返回 {index: observation} 字典
    """
    results = {}

    def _do_search(idx, query, tool_type):
        if tool_type == "web" and use_serper:
            return idx, serper_search(query, num_results=3)
        else:
            return idx, kb.search(query, top_k=SEARCH_TOP_K)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_do_search, idx, q, t) for idx, q, t in queries_with_idx]
        for future in as_completed(futures):
            idx, obs = future.result()
            results[idx] = obs[:400]  # 截断防止上下文爆炸

    return results


# =================================─────────────
# 6. 批量并行 ReAct Agent 核心循环
# =================================─────────────
_PANGU_NOISE = re.compile(r"\[unused(?:\d+)\]|<think>|</think>|<s>|</s>")


def run_batch_react_agents(test_data, llm, sampling_params, kb, max_steps=MAX_STEPS):
    """
    并行运行所有样本的 ReAct Agent。
    核心思想：按 step 驱动，每一步将所有未完成样本的 prompt 打包成 batch 提交 vLLM。
    """
    total = len(test_data)
    use_serper = is_serper_available()
    max_scratchpad_chars = 10000  # 约 2500 token，足够保留最近 3-4 轮交互

    # 每条样本的状态
    states = []
    for i, item in enumerate(test_data):
        states.append({
            "idx": i,
            "question": item["question"],
            "scratchpad": "",
            "trajectory": [],
            "answer": "",
            "finished": False,
        })

    # ====== 预热阶段：无条件用原始问题执行第一跳检索，把结果预填入 scratchpad ======
    # 这样模型从第一步开始就已经有了上下文，大幅减少 FormatError 导致的浪费
    print(f"\n  [Pre-fetch] 对 {total} 条问题无条件执行第一跳本地检索...")
    all_questions = [s["question"] for s in states]
    # 批量编码 + FAISS 检索
    query_vectors = kb.embed_model.encode(all_questions, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(query_vectors)
    distances, indices = kb.index.search(query_vectors, SEARCH_TOP_K)

    for i in range(total):
        obs_parts = []
        for score, idx in zip(distances[i], indices[i]):
            if 0 <= idx < len(kb.corpus_data):
                doc = kb.corpus_data[idx]
                obs_parts.append(f"[{doc['title']}]: {doc['text'][:300]}")
        obs = "\n".join(obs_parts)[:400]
        states[i]["scratchpad"] = f"\nThought: I will first search for information related to the question.\nAction: Search[{states[i]['question'][:80]}]\nObservation: {obs}\n"
        states[i]["trajectory"].append({
            "step": 0,
            "thought": "Pre-fetch: initial knowledge base search",
            "action": f"Search[{states[i]['question'][:80]}]",
            "observation": obs
        })

    print(f"  [Pre-fetch] 完成！所有样本已预填初始检索结果。")

    for step in range(1, max_steps + 1):
        # 收集本轮未完成的样本
        active_indices = [i for i, s in enumerate(states) if not s["finished"]]
        if not active_indices:
            break

        print(f"\n  [Step {step}/{max_steps}] 活跃样本数: {len(active_indices)}")

        # 6.1 批量构建 prompts（自动截断过长 scratchpad 防止超出 max_model_len）
        prompts = []
        for i in active_indices:
            s = states[i]
            # 如果 scratchpad 过长，只保留最后 N 个字符（最近几轮）
            scratchpad = s["scratchpad"]
            if len(scratchpad) > max_scratchpad_chars:
                scratchpad = "...[earlier steps truncated]...\n" + scratchpad[-max_scratchpad_chars:]
            prompt = REACT_PROMPT_TEMPLATE.format(
                question=s["question"],
                scratchpad=scratchpad,
                max_steps=max_steps
            )
            prompts.append(prompt)

        # 6.2 批量提交 vLLM 并行生成（核心加速点！）
        outputs = llm.generate(prompts, sampling_params)

        # 6.3 解析生成结果，分类处理
        search_tasks = []  # [(state_idx, search_query, tool_type)]

        for batch_idx, global_idx in enumerate(active_indices):
            s = states[global_idx]
            raw_output = outputs[batch_idx].outputs[0].text if outputs[batch_idx].outputs else ""
            raw_output = _PANGU_NOISE.sub("", raw_output).strip()

            # 解析 Finish[answer]
            finish_match = re.search(r"Finish\[(.+?)\]", raw_output)
            if finish_match:
                answer = finish_match.group(1).strip().strip(".,'\""" ")
                thought_match = re.search(r"Thought:\s*(.+?)(?=Action:)", raw_output, re.DOTALL)
                thought = thought_match.group(1).strip() if thought_match else ""
                s["trajectory"].append({"step": step, "thought": thought, "action": f"Finish[{answer}]", "observation": ""})
                s["answer"] = answer
                s["finished"] = True
                continue

            # 解析 WebSearch[query]（优先匹配，因为 "WebSearch" 包含 "Search"）
            websearch_match = re.search(r"WebSearch\[(.+?)\]", raw_output)
            if websearch_match:
                search_query = websearch_match.group(1).strip()
                thought_match = re.search(r"Thought:\s*(.+?)(?=Action:)", raw_output, re.DOTALL)
                thought = thought_match.group(1).strip() if thought_match else ""
                s["_pending_thought"] = thought
                s["_pending_query"] = search_query
                s["_pending_tool"] = "web"
                search_tasks.append((global_idx, search_query, "web"))
                continue

            # 解析 Search[query]（本地知识库检索）
            search_match = re.search(r"Search\[(.+?)\]", raw_output)
            if search_match:
                search_query = search_match.group(1).strip()
                thought_match = re.search(r"Thought:\s*(.+?)(?=Action:)", raw_output, re.DOTALL)
                thought = thought_match.group(1).strip() if thought_match else ""
                s["_pending_thought"] = thought
                s["_pending_query"] = search_query
                # 策略：第一次 Search 走本地，后续如果是重复搜索或本地已失败过，自动升级为 WebSearch
                if step >= 2 and use_serper:
                    s["_pending_tool"] = "web"
                    search_tasks.append((global_idx, search_query, "web"))
                else:
                    s["_pending_tool"] = "local"
                    search_tasks.append((global_idx, search_query, "local"))
            else:
                # 格式错误兜底：先从 raw_output 中尝试多种模式提取答案
                s["trajectory"].append({"step": step, "thought": raw_output[:200], "action": "FormatError", "observation": ""})
                
                # 策略 A: 匹配 "the answer is X" / "Final Answer: X" / "So the answer is X" 等
                answer_patterns = [
                    re.compile(r"(?:the answer is|final answer:?|answer:)\s*['\"]?([^'\"\.\n,]+)", re.IGNORECASE),
                    re.compile(r"(?:so|therefore),?\s+(?:the answer is|it is)\s*['\"]?([^'\"\.\n,]+)", re.IGNORECASE),
                ]
                extracted = ""
                for pat in answer_patterns:
                    m = pat.search(raw_output)
                    if m:
                        extracted = m.group(1).strip().strip(".,'\""" ")
                        break
                
                if not extracted:
                    # 策略 B: 取第一个非废话短行
                    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
                    for line in lines:
                        clean_line = re.sub(r"^(Thought:|Action:|Step \d+:)\s*", "", line, flags=re.IGNORECASE).strip()
                        if clean_line and len(clean_line.split()) <= 6 and not re.match(r"^(I need|Let me|Okay|Wait|Note|The question)", clean_line, re.IGNORECASE):
                            extracted = clean_line.strip(".,'\""" ")
                            break
                
                if extracted and len(extracted.split()) <= 8:
                    s["answer"] = extracted
                    s["finished"] = True
                else:
                    # FormatError 兜底：有 Serper 时直接用 WebSearch（远比本地 FAISS 好）
                    s["_pending_thought"] = "I need to search for more information."
                    s["_pending_query"] = s["question"]
                    if use_serper:
                        s["_pending_tool"] = "web"
                        search_tasks.append((global_idx, s["question"], "web"))
                    else:
                        s["_pending_tool"] = "local"
                        search_tasks.append((global_idx, s["question"], "local"))

        # 6.4 并行执行所有检索任务（多线程，区分 local/web）
        if search_tasks:
            local_count = sum(1 for _, _, t in search_tasks if t == "local")
            web_count = sum(1 for _, _, t in search_tasks if t == "web")
            print(f"  [Step {step}] 并行检索 {len(search_tasks)} 条 (本地FAISS: {local_count}, WebSearch: {web_count})...")
            observations = batch_search(search_tasks, use_serper, kb)

            # 6.5 将 observation 写回各样本的 scratchpad
            for global_idx, query, tool_type in search_tasks:
                s = states[global_idx]
                obs = observations.get(global_idx, "No results found.")
                thought = s.pop("_pending_thought", "")
                search_q = s.pop("_pending_query", query)
                tool_name = "WebSearch" if s.pop("_pending_tool", "local") == "web" else "Search"
                s["trajectory"].append({"step": step, "thought": thought, "action": f"{tool_name}[{search_q}]", "observation": obs})
                s["scratchpad"] += f"\nThought: {thought}\nAction: {tool_name}[{search_q}]\nObservation: {obs}\n"

    # ======= 兜底：对未完成的样本，基于已有 scratchpad 强制生成最终答案 =======
    unfinished = [i for i, s in enumerate(states) if not s["finished"] and s["scratchpad"].strip()]
    if unfinished:
        print(f"\n  [Final Fallback] {len(unfinished)} 条样本未在步数内 Finish，正在强制生成答案...")
        fallback_prompts = []
        for i in unfinished:
            s = states[i]
            # 从 trajectory 中提取最佳 observation（排除超时/错误的，优先取内容最丰富的）
            best_obs = ""
            for t in s["trajectory"]:
                obs = t.get("observation", "").strip()
                if obs and "error" not in obs.lower() and "timed out" not in obs.lower() and len(obs) > len(best_obs):
                    best_obs = obs

            # 使用 Few-shot 格式 + 直接给出 observation 内容，让模型从中提取答案
            fallback_prompt = (
                f"Extract the answer from the search results. Output ONLY the short answer (1-5 words). No explanation.\n\n"
                f"Example:\n"
                f"Question: Who directed Interstellar?\n"
                f"Search Results: [Christopher Nolan]: British-American film director known for The Dark Knight, Interstellar...\n"
                f"Answer: Christopher Nolan\n\n"
                f"Example:\n"
                f"Question: When was UC Berkeley founded?\n"
                f"Search Results: [University of California, Berkeley]: Founded in 1868, it is the flagship...\n"
                f"Answer: 1868\n\n"
                f"Now extract:\n"
                f"Question: {s['question']}\n"
                f"Search Results: {best_obs[:500]}\n"
                f"Answer:"
            )
            fallback_prompts.append(fallback_prompt)

        # 用特殊的 sampling params（短输出 + 换行即停）
        fallback_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,
            stop=["\n", "Question:", "Example"],
            repetition_penalty=1.1,
        )
        fallback_outputs = llm.generate(fallback_prompts, fallback_params)

        for batch_idx, global_idx in enumerate(unfinished):
            s = states[global_idx]
            raw = fallback_outputs[batch_idx].outputs[0].text if fallback_outputs[batch_idx].outputs else ""
            raw = _PANGU_NOISE.sub("", raw).strip().strip(".,'\""" ]")
            # 去除常见的前缀垃圾
            raw = re.sub(r"^(answer:|the answer is|output:)\s*", "", raw, flags=re.IGNORECASE).strip()
            s["answer"] = raw if raw else ""
            s["finished"] = True
            s["trajectory"].append({"step": max_steps + 1, "thought": "Forced final answer extraction", "action": f"Finish[{raw}]", "observation": ""})

    # 返回最终结果
    results = []
    for s in states:
        results.append({
            "answer": s["answer"],
            "trajectory": s["trajectory"],
        })
    return results


# =================================─────────────
# 7. 主执行流
# =================================─────────────
def parse_args():
    parser = argparse.ArgumentParser(description="并行 ReAct Agent + Serper API 多跳问答")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_PATH)
    parser.add_argument("--corpus_json", type=str, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--input_json", type=str, required=True, help="测试集路径")
    parser.add_argument("--output_json", type=str, default="predictions_agent.json", help="预测输出路径")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Agent 最大迭代步数")
    parser.add_argument("--top_k", type=int, default=SEARCH_TOP_K, help="本地 FAISS Top-K")
    parser.add_argument("--serper_key", type=str, default=None, help="Serper API Key")
    return parser.parse_args()


def main():
    args = parse_args()
    global SEARCH_TOP_K, MAX_STEPS, SERPER_API_KEY
    SEARCH_TOP_K = args.top_k
    MAX_STEPS = args.max_steps
    if args.serper_key:
        SERPER_API_KEY = args.serper_key

    if is_serper_available():
        print("[+] Serper API Key 已配置，使用 Google 搜索！")
    else:
        print("[!] 未检测到 Serper API Key，回退到本地 FAISS。")

    # 加载本地向量库（Agent 优先使用本地 FAISS 检索，WebSearch 作为补充）
    kb = VectorKnowledgeBase(args.embedding_model, device="cpu")
    if not kb.load_index():
        print("[*] 首次构建 FAISS 索引...")
        kb.build_index_from_json(args.corpus_json)

    # 加载 vLLM
    print(f"[*] 初始化 vLLM (TP={args.tensor_parallel_size}) ...")
    llm = LLM(
        model=args.llm_model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=250,
        repetition_penalty=1.1,
        stop=["Observation:", "\nObservation"],
    )

    # 加载测试数据
    print(f"[*] 加载测试集: {args.input_json}")
    with open(args.input_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    total = len(test_data)
    print(f"[+] 共 {total} 条样本，开始并行 ReAct Agent 推理...")

    # 并行运行
    t0 = time.time()
    agent_results = run_batch_react_agents(test_data, llm, sampling_params, kb, max_steps=args.max_steps)
    elapsed = time.time() - t0

    # 组装输出
    predictions_json = []
    for i, item in enumerate(test_data):
        res = agent_results[i]
        predictions_json.append({
            "id": item.get("id", ""),
            "question": item["question"],
            "gold_answer": item.get("answer", ""),
            "prediction": res["answer"],
            "type": item.get("type", ""),
            "trajectory": res["trajectory"],
        })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions_json, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[+] 并行 ReAct Agent 多跳问答完成！")
    print(f"[+] 检索模式: {'Serper (Google)' if is_serper_available() else 'Local FAISS'}")
    print(f"[+] 总耗时: {elapsed:.1f}s，平均每条: {elapsed/total:.1f}s")
    print(f"[+] 预测结果已保存至: {os.path.abspath(args.output_json)}")
    print("=" * 60)


if __name__ == "__main__":
    main()