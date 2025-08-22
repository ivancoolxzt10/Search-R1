import json  # JSON处理库
import os  # 操作系统相关库
import warnings  # 警告处理库
from typing import List, Dict  # 类型注解
import functools  # 高阶函数工具库
from tqdm import tqdm  # 进度条库
from multiprocessing import Pool  # 多进程库
import faiss  # 向量检索库
import torch  # PyTorch深度学习库
import numpy as np  # 数值计算库
from transformers import AutoConfig, AutoTokenizer, AutoModel  # Transformers库
import argparse  # 命令行参数解析库
import datasets  # HuggingFace数据集库


# 功能说明：加载语料库，支持jsonl格式并多进程加速。
def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
            'json', 
            data_files=corpus_path,
            split="train",
            num_proc=4)
    return corpus

# 功能说明：读取jsonl文件，返回数据列表。
def read_jsonl(file_path):
    data = []
    
    with open(file_path, "r") as f:
        readin = f.readlines()
        for line in readin:
            data.append(json.loads(line))
    return data


# 功能说明：根据文档索引列表从语料库加载文档。
def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results


# 功能说明：加载检索模型和分词器，支持FP16。
def load_model(
        model_path: str, 
        use_fp16: bool = False
    ):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


# 功能说明：池化方法，支持mean、cls、pooler。
def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask = None,
        pooling_method = "mean"
    ):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


# 功能说明：Encoder类用于文本编码为向量，支持多种检索模型。
class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path,
                                                use_fp16=use_fp16)

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # 处理不同模型的输入前缀
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5模型特殊处理
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb


# BaseRetriever类：检索器基类，定义通用接口。
class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

        # self.cache_save_path = os.path.join(config.save_dir, 'retrieval_cache.json')

    def _search(self, query: str, num: int, return_score:bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.
        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)
        """
        pass

    def _batch_search(self, query_list, num, return_score):
        pass

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)
    
    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)


# BM25Retriever类：基于Pyserini的BM25检索器，支持批量检索和分数返回。
class BM25Retriever(BaseRetriever):
    r"""BM25 retriever based on pre-built pyserini index."""

    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher  # 导入Pyserini Lucene检索器
        self.searcher = LuceneSearcher(self.index_path)  # 初始化检索器
        self.contain_doc = self._check_contain_doc()  # 检查索引是否包含文档内容
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)  # 若无内容则加载语料库
        self.max_process_num = 8  # 最大进程数

    def _check_contain_doc(self):
        r"""检查索引是否包含文档内容。"""
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score = False) -> List[Dict[str, str]]:
        """
        单条query检索，返回topk文档及分数。
        参数：
            query: 查询文本
            num: 返回文档数量
            return_score: 是否返回分数
        返回：
            results: 检索到的文档列表
            scores: 检索分数列表（可选）
        """
        # 单条query检索，返回topk文档及分数
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)  # 检索
        if len(hits) < 1:
            if return_score:
                return [],[]  # 无结果返回空
            else:
                return []
        scores = [hit.score for hit in hits]  # 获取分数
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')  # 检索结果不足
        else:
            hits = hits[:num]
        if self.contain_doc:
            # 索引包含内容，直接解析
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in hits]
            results = [{'title': content.split("\n")[0].strip("\""), 
                        'text': "\n".join(content.split("\n")[1:]),
                        'contents': content} for content in all_contents]
        else:
            # 否则从语料库加载
            results = load_docs(self.corpus, [hit.docid for hit in hits])
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list, num: int = None, return_score = False):
        # 批量检索，每条query调用_search
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num,True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

# DenseRetriever类：基于Faiss的稠密检索器，支持GPU加速、批量检索和分数返回。
class DenseRetriever(BaseRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)  # 加载Faiss索引文件
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()  # GPU克隆选项
            co.useFloat16 = True  # 使用FP16加速
            co.shard = True  # 分片到多GPU
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)  # 将CPU索引转为GPU索引
        self.corpus = load_corpus(self.corpus_path)  # 加载语料库
        self.encoder = Encoder(
             model_name = self.retrieval_method, 
             model_path = config.retrieval_model_path,
             pooling_method = config.retrieval_pooling_method,
             max_length = config.retrieval_query_max_length,
             use_fp16 = config.retrieval_use_fp16
            )  # 初始化编码器
        self.topk = config.retrieval_topk  # topk参数
        self.batch_size = self.config.retrieval_batch_size  # 批量大小

    def _search(self, query: str, num: int = None, return_score = False):
        """
        单条query检索，返回topk文档及分数。
        参数：
            query: 查询文本
            num: 返回文档数量
            return_score: 是否返回分数
        返回：
            results: 检索到的文档列表
            scores: 检索分数列表（可选）
        """
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)  # 编码query为向量
        scores, idxs = self.index.search(query_emb, k=num)  # Faiss检索
        idxs = idxs[0]  # 取第一个batch
        scores = scores[0]
        results = load_docs(self.corpus, idxs)  # 加载文档内容
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score = False):
        """
        批量检索，支持大规模并行。
        参数：
            query_list: 查询文本列表
            num: 返回文档数量
            return_score: 是否返回分数
        返回：
            results: 检索到的文档列表
            scores: 检索分数列表（可选）
        """
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        batch_size = self.batch_size
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + batch_size]  # 当前批次
            batch_emb = self.encoder.encode(query_batch)  # 批量编码
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)  # Faiss批量检索
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            flat_idxs = sum(batch_idxs, [])  # 展平索引
            batch_results = load_docs(self.corpus, flat_idxs)  # 加载文档内容
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]  # 按批次分组
            scores.extend(batch_scores)
            results.extend(batch_results)
        if return_score:
            return results, scores
        else:
            return results

# get_available_gpu_memory函数：获取所有GPU的剩余显存信息。
def get_available_gpu_memory():
    memory_info = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory  # 总显存
        allocated_memory = torch.cuda.memory_allocated(i)  # 已分配显存
        free_memory = total_memory - allocated_memory  # 剩余显存
        memory_info.append((i, free_memory / 1e9))  # 转为GB
    return memory_info

# get_retriever函数：根据配置自动选择检索器类型。
def get_retriever(config):
    r"""根据配置选择BM25或Dense检索器。"""
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)

# get_dataset函数：根据配置加载数据集。
def get_dataset(config):
    """根据配置加载jsonl数据集。"""
    split_path = os.path.join(config.dataset_path, f'{config.data_split}.jsonl')
    return read_jsonl(split_path)

# 主流程入口，负责参数解析、数据加载、检索器初始化、批量检索。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Retrieval")
    # 基本参数
    parser.add_argument('--retrieval_method', type=str)  # 检索方法
    parser.add_argument('--retrieval_topk', type=int, default=10)  # topk
    parser.add_argument('--index_path', type=str, default=None)  # 索引路径
    parser.add_argument('--corpus_path', type=str)  # 语料库路径
    parser.add_argument('--dataset_path', default=None, type=str)  # 数据集路径
    parser.add_argument('--faiss_gpu', default=True, type=bool)  # 是否用GPU
    parser.add_argument('--data_split', default="train", type=str)  # 数据集分割
    parser.add_argument('--retrieval_model_path', type=str, default=None)  # 检索模型路径
    parser.add_argument('--retrieval_pooling_method', default='mean', type=str)  # 池化方法
    parser.add_argument('--retrieval_query_max_length', default=256, type=str)  # 最大长度
    parser.add_argument('--retrieval_use_fp16', action='store_true', default=False)  # 是否用FP16
    parser.add_argument('--retrieval_batch_size', default=512, type=int)  # 批量大小
    args = parser.parse_args()
    # 构造索引路径
    args.index_path = os.path.join(args.index_path, f'{args.retrieval_method}_Flat.index') if args.retrieval_method != 'bm25' else os.path.join(args.index_path, 'bm25')
    # 加载数据集
    all_split = get_dataset(args)
    input_query = [sample['question'] for sample in all_split[:512]]  # 取前512条问题
    # 初始化检索器并检索
    retriever = get_retriever(args)
    print('Start Retrieving ...')    
    results, scores = retriever.batch_search(input_query, return_score=True)
    # from IPython import embed
    # embed()

# 文件整体功能总结：
# retrieval.py 实现了通用检索模块，支持BM25和稠密检索，涵盖语料加载、模型编码、索引检索、批量处理、分数返回等，
# 便于大规模问答、信息检索、RAG等场景的高效数据流和模型集成。
