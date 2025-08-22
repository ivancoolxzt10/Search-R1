# pip install -U sentence-transformers
import os  # 操作系统相关库
import re  # 正则表达式库
import argparse  # 命令行参数解析库
from dataclasses import dataclass, field  # 数据类工具
from typing import List, Optional  # 类型注解
from collections import defaultdict  # 默认字典容器

import torch  # PyTorch深度学习库
import numpy as np  # 数值计算库
from fastapi import FastAPI  # Web框架
from pydantic import BaseModel  # 数据模型校验
from sentence_transformers import CrossEncoder  # 句子交叉编码器

from retrieval_server import get_retriever, Config as RetrieverConfig  # 检索器工具
from rerank_server import SentenceTransformerCrossEncoder  # 重排序器工具

app = FastAPI()

# 功能说明：格式化标题和内容，便于统一输出。
def convert_title_format(text):
    # 用正则提取标题和内容
    match = re.match(r'\(Title:\s*([^)]+)\)\s*(.+)', text, re.DOTALL)
    if match:
        title, content = match.groups()
        return f'"{title}"\n{content}'
    else:
        return text

# ----------- Combined Request Schema -----------
class SearchRequest(BaseModel):
    queries: List[str]  # 查询列表
    topk_retrieval: Optional[int] = 10  # 检索topk
    topk_rerank: Optional[int] = 3  # 重排序topk
    return_scores: bool = False  # 是否返回分数

# ----------- Reranker Config Schema -----------
@dataclass
class RerankerArguments:
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")

# 获取重排序器实例
def get_reranker(config):
    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")

# ----------- Endpoint -----------
@app.post("/retrieve")
def search_endpoint(request: SearchRequest):
    """
    FastAPI接口：检索+重排序一体化服务。
    步骤：
    1. 检索topk文档
    2. 对检索结果进行重排序
    3. 格式化输出
    """
    # Step 1: Retrieve documents
    retrieved_docs = retriever.batch_search(
        query_list=request.queries,
        num=request.topk_retrieval,
        return_score=False
    )
    # Step 2: Rerank
    reranked = reranker.rerank(request.queries, retrieved_docs)
    # Step 3: Format response
    response = []
    for i, doc_scores in reranked.items():
        doc_scores = doc_scores[:request.topk_rerank]
        if request.return_scores:
            combined = []
            for doc, score in doc_scores:
                combined.append({"document": convert_title_format(doc), "score": score})
            response.append(combined)
        else:
            response.append([convert_title_format(doc) for doc, _ in doc_scores])
    return {"result": response}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    # retriever
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--retrieval_topk", type=int, default=10, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    # reranker
    parser.add_argument("--reranking_topk", type=int, default=3, help="Number of reranked passages for one query.")
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2", help="Path of the reranker model.")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="Batch size for the reranker inference.")
    args = parser.parse_args()
    # ----------- Load Retriever and Reranker -----------
    retriever_config = RetrieverConfig(
        retrieval_method = args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.retrieval_topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    retriever = get_retriever(retriever_config)
    reranker_config = RerankerArguments(
        rerank_topk = args.reranking_topk,
        rerank_model_name_or_path = args.reranker_model,
        batch_size = args.reranker_batch_size,
    )
    reranker = get_reranker(reranker_config)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 文件整体功能总结：
# retrieval_rerank_server.py 实现了检索+重排序一体化服务，支持批量查询、分数返回、格式化输出，便于RAG、问答等场景的高效集成。
