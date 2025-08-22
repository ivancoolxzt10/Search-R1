import argparse  # 命令行参数解析库
from collections import defaultdict  # 默认字典容器
from typing import Optional  # 类型注解
from dataclasses import dataclass, field  # 数据类工具

from sentence_transformers import CrossEncoder  # 句子交叉编码器
import torch  # PyTorch深度学习库
from transformers import HfArgumentParser  # HuggingFace参数解析器
import numpy as np  # 数值计算库

import uvicorn  # ASGI服务器
from fastapi import FastAPI  # Web框架
from pydantic import BaseModel  # 数据模型校验

# 基础交叉编码器类，支持批量重排序。
class BaseCrossEncoder:
    def __init__(self, model, batch_size=32, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.model.to(device)

    def _passage_to_string(self, doc_item):
        """
        将文档字典转为字符串，格式：(Title: 标题) 正文
        """
        if "document" not in doc_item:
            content = doc_item['contents']
        else:
            content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])

        return f"(Title: {title}) {text}"

    def rerank(self, 
               queries: list[str], 
               documents: list[list[dict]]):
        """
        批量重排序接口。
        queries: 查询列表
        documents: 每个查询对应的文档列表（嵌套字典）
        返回：每个查询对应的文档及分数，按分数降序排列
        """
        assert len(queries) == len(documents)

        pairs = []
        qids = []
        for qid, query in enumerate(queries):
            for document in documents:
                for doc_item in document:
                    doc = self._passage_to_string(doc_item)
                    pairs.append((query, doc))
                    qids.append(qid)

        scores = self._predict(pairs)
        query_to_doc_scores = defaultdict(list)

        assert len(scores) == len(pairs) == len(qids)
        for i in range(len(pairs)):
            query, doc = pairs[i]
            score = scores[i] 
            qid = qids[i]
            query_to_doc_scores[qid].append((doc, score))

        sorted_query_to_doc_scores = {}
        for query, doc_scores in query_to_doc_scores.items():
            sorted_query_to_doc_scores[query] = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return sorted_query_to_doc_scores

    def _predict(self, pairs: list[tuple[str, str]]):
        """
        预测分数，需子类实现。
        """
        raise NotImplementedError

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        """
        加载模型，需子类实现。
        """
        raise NotImplementedError


# SentenceTransformer交叉编码器实现
class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    def __init__(self, model, batch_size=32, device="cuda"):
        super().__init__(model, batch_size, device)

    def _predict(self, pairs: list[tuple[str, str]]):
        """
        使用CrossEncoder模型预测分数。
        """
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scores = scores.tolist() if isinstance(scores, torch.Tensor) or isinstance(scores, np.ndarray) else scores
        return scores

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        """
        加载CrossEncoder模型。
        """
        model = CrossEncoder(model_name_or_path)
        return cls(model, **kwargs)


# FastAPI请求体定义
class RerankRequest(BaseModel):
    queries: list[str]
    documents: list[list[dict]]
    rerank_topk: Optional[int] = None
    return_scores: bool = False


# 重排序参数数据类
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


app = FastAPI()

@app.post("/rerank")
def rerank_endpoint(request: RerankRequest):
    """
    FastAPI接口：批量重排序。
    输入格式：
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "documents": [[doc_item_1, ..., doc_item_k], [doc_item_1, ..., doc_item_k]],
      "rerank_topk": 3,
      "return_scores": true
    }
    """
    if not request.rerank_topk:
        request.rerank_topk = config.rerank_topk  # 默认topk

    # 批量重排序
    query_to_doc_scores = reranker.rerank(request.queries, request.documents)

    # 格式化返回结果
    resp = []
    for _, doc_scores in query_to_doc_scores.items():
        doc_scores = doc_scores[:request.rerank_topk]
        if request.return_scores:
            combined = [] 
            for doc, score in doc_scores:
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append([doc for doc, _ in doc_scores])
    return {"result": resp}


if __name__ == "__main__":
    # 1) 构建配置（可通过命令行参数或环境变量解析）。
    parser = HfArgumentParser((RerankerArguments))
    config = parser.parse_args_into_dataclasses()[0]

    # 2) 实例化全局重排序器，避免重复加载
    reranker = get_reranker(config)
    
    # 3) 启动服务，默认监听 http://0.0.0.0:6980
    uvicorn.run(app, host="0.0.0.0", port=6980)
