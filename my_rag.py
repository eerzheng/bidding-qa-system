#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
招投标智能问答系统 - 优化版（解决 Recall@3 过低问题）
改进点：
1. 重排序时不再对候选块做上下文拼接，直接对原始 chunk 打分
2. 降低向量候选数（80 -> 40），减少噪声
3. 最终返回的每个 chunk 会动态拼接前后块，保证给 LLM 的上下文完整
"""

import sys
import os
import pickle
import requests
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

# ================== 配置 ==================
SILICONFLOW_API_KEY = "sk-noqlfpczryltqmqbmrfhlxbikquaaawrafyylzdsymulrvsp"   # 请替换为有效key
CHROMA_DB_PATH = "./processed_data/chroma_db"
COLLECTION_NAME = "bidding_law_kb"
DOCS_LIST_PATH = "./processed_data/documents_list.pkl"
METAS_LIST_PATH = "./processed_data/metadatas_list.pkl"

VECTOR_TOP_K = 40          # 向量检索候选数（原80，减少干扰）
FINAL_TOP_K = 5            # 最终返回给 LLM 的 chunk 数（原6）
USE_RERANKER = True
CONTEXT_WINDOW = 1         # 最终为每个 chunk 拼接前后多少块（仅用于最终上下文，不参与重排序）

SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
TEMPERATURE = 0.3          # 降低随机性，更依赖事实
MAX_TOKENS = 800
MAX_CONTEXT_LEN = 3500

# ================== 初始化 ==================
def init_chroma():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="shibing624/text2vec-base-chinese"
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    print(f"✅ Chroma 加载成功，共 {collection.count()} 条")
    return collection

def load_docs_and_metas():
    with open(DOCS_LIST_PATH, "rb") as f:
        docs = pickle.load(f)
    with open(METAS_LIST_PATH, "rb") as f:
        metas = pickle.load(f)
    print(f"✅ 文档列表加载成功，共 {len(docs)} 条")
    return docs, metas

# 重排序模型
reranker = None
if USE_RERANKER:
    try:
        print("正在加载重排序模型...")
        reranker = CrossEncoder('BAAI/bge-reranker-base')
        print("✅ 重排序模型加载成功")
    except Exception as e:
        print(f"⚠️ 重排序模型加载失败：{e}，将不使用重排序")
        USE_RERANKER = False

# ================== 检索（优化版：无噪声重排序）==================
def retrieve(question, collection, documents, metadatas):
    # 1. 向量检索获取候选索引
    results = collection.query(query_texts=[question], n_results=VECTOR_TOP_K)
    ids = results['ids'][0]               # ['chunk_0', 'chunk_1', ...]
    indices = [int(id.split('_')[1]) for id in ids]
    distances = results['distances'][0]
    # 按距离排序（由近到远）
    sorted_idx = sorted(range(len(indices)), key=lambda i: distances[i])
    candidate_indices = [indices[i] for i in sorted_idx]

    # 2. 重排序（直接对原始 chunk 打分，不做上下文拼接）
    if USE_RERANKER and reranker is not None:
        # 直接使用原始文档内容进行重排序
        candidate_docs = [documents[i] for i in candidate_indices]
        pairs = [[question, doc] for doc in candidate_docs]
        scores = reranker.predict(pairs)
        # 按重排序分数降序排列
        rerank_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        final_indices = [candidate_indices[i] for i in rerank_idx[:FINAL_TOP_K]]
    else:
        final_indices = candidate_indices[:FINAL_TOP_K]

    # 3. 为最终选中的每个 chunk 拼接前后块（提高上下文连贯性，但不再参与排序）
    final_docs = []
    final_metas = []
    for idx in final_indices:
        start = max(0, idx - CONTEXT_WINDOW)
        end = min(len(documents) - 1, idx + CONTEXT_WINDOW)
        context_blocks = [documents[i] for i in range(start, end+1)]
        combined = "\n".join(context_blocks)
        final_docs.append(combined)
        final_metas.append(metadatas[idx])   # 元数据仍使用原始块的

    # 4. 构建上下文字符串（供 LLM 使用）
    context_parts = []
    for doc, meta in zip(final_docs, final_metas):
        source_info = f"[来源：{meta.get('source', '未知')}"
        if meta.get('type'):
            source_info += f" 类型：{meta['type']}"
        if meta.get('title'):
            source_info += f" 章节：{meta['title']}"
        if meta.get('start_page'):
            source_info += f" 页码：{meta['start_page']}"
        if meta.get('article'):
            source_info += f" 法条：{meta['article']}"
        source_info += "]"
        context_parts.append(f"{source_info}\n{doc}")
    context = "\n\n".join(context_parts)
    if len(context) > MAX_CONTEXT_LEN:
        context = context[:MAX_CONTEXT_LEN] + "..."
    return context, final_metas

# ================== 答案生成 ==================
def generate_answer(collection, documents, metadatas, question):
    context, metas = retrieve(question, collection, documents, metadatas)
    if not context:
        return "未找到相关内容，请换个问题试试。", ""

    system_prompt = """你是招投标领域的专业顾问。请严格遵守：
1. 仅根据【知识库上下文】回答，不编造任何内容。
2. 使用数字列表分条陈述，每条后注明来源。
3. 如果上下文中有案例，末尾加【相关案例】。
4. 信息不足则回答“根据现有资料无法确定”。"""

    user_prompt = f"【知识库上下文】\n{context}\n\n【用户问题】\n{question}\n\n【回答】"

    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    try:
        resp = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        return answer or "未找到相关内容", context
    except Exception as e:
        return f"API调用失败：{e}", context

# ================== 主程序 ==================
def main():
    if SILICONFLOW_API_KEY == "sk-noqlfpczryltqmqbmrfhlxbikquaaawrafyylzdsymulrvsp":
        print("⚠️ 请先设置有效的硅基流动 API Key（替换脚本中的 SILICONFLOW_API_KEY）")
        sys.exit(1)
    collection = init_chroma()
    documents, metadatas = load_docs_and_metas()
    print("\n" + "="*60)
    print("招投标智能问答系统 - 优化版（直接重排序，无噪声）")
    print("输入问题，输入 exit 退出")
    print("="*60 + "\n")
    while True:
        q = input(">>> ").strip()
        if q.lower() in ("exit","quit"): break
        if not q: continue
        print("正在检索并生成答案...")
        ans, ctx = generate_answer(collection, documents, metadatas, q)
        print("\n【检索依据】\n", ctx[:1200])
        print("\n【回答】\n", ans)
        print("-"*60)

if __name__ == "__main__":
    main()