import os  # 操作系统相关库
import faiss  # 向量检索库
import json  # JSON处理库
import warnings  # 警告处理库
import numpy as np  # 数值计算库
from typing import cast, List, Dict  # 类型注解
import shutil  # 文件操作库
import subprocess  # 子进程管理库
import argparse  # 命令行参数解析库
import torch  # PyTorch深度学习库
from tqdm import tqdm  # 进度条库
# from LongRAG.retriever.utils import load_model, load_corpus, pooling
import datasets  # HuggingFace数据集库
from transformers import AutoTokenizer, AutoModel, AutoConfig  # Transformers库


# 功能说明：加载预训练模型和分词器。
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


# 功能说明：根据指定方法对模型输出进行池化，得到句向量。
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


# 功能说明：加载语料库。
def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
            'json', 
            data_files=corpus_path,
            split="train",
            num_proc=4)
    return corpus


# 功能说明：索引构建主类，支持BM25和Dense索引。
class Index_Builder:
    r"""A tool class used to build an index used in retrieval.
    检索索引构建工具类，支持稠密向量和BM25索引。
    """
    def __init__(
            self, 
            retrieval_method,
            model_path,
            corpus_path,
            save_dir,
            max_length,
            batch_size,
            use_fp16,
            pooling_method,
            faiss_type=None,
            embedding_path=None,
            save_embedding=False,
            faiss_gpu=False
        ):
        # 初始化参数
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else 'Flat'
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu

        self.gpu_num = torch.cuda.device_count()  # GPU数量
        # 准备保存目录
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")

        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")

        self.corpus = load_corpus(self.corpus_path)  # 加载语料库

        print("Finish loading...")

    @staticmethod
    def _check_dir(dir_path):
        r"""检查目录是否存在且为空。"""
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        r"""根据检索方法构建索引。"""
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """基于Pyserini库构建BM25索引。"""

        # 将jsonl文件放入指定文件夹
        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)

        shutil.copyfile(self.corpus_path, temp_file_path)
        
        print("Start building bm25 index...")
        pyserini_args = ["--collection", "JsonCollection",
                         "--input", temp_dir,
                         "--index", self.save_dir,
                         "--generator", "DefaultLuceneDocumentGenerator",
                         "--threads", "1"]
       
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)

        shutil.rmtree(temp_dir)
        
        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        """加载memmap格式的embedding。"""
        all_embeddings = np.memmap(
                embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(corpus_size, hidden_size)
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        """保存embedding到memmap文件。"""
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype
        )
        length = all_embeddings.shape[0]
        # add in batch
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i: j] = all_embeddings[i: j]
        else:
            memmap[:] = all_embeddings

    def encode_all(self):
        """批量编码所有语料，得到embedding。"""
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.batch_size = self.batch_size * self.gpu_num

        all_embeddings = []

        for start_idx in tqdm(range(0, len(self.corpus), self.batch_size), desc='Inference Embeddings:'):

            # batch_data_title = self.corpus[start_idx:start_idx+self.batch_size]['title']
            # batch_data_text = self.corpus[start_idx:start_idx+self.batch_size]['text']
            # batch_data = ['"' + title + '"\n' + text for title, text in zip(batch_data_title, batch_data_text)]
            batch_data = self.corpus[start_idx:start_idx+self.batch_size]['contents']

            if self.retrieval_method == "e5":
                batch_data = [f"passage: {doc}" for doc in batch_data]

            inputs = self.tokenizer(
                        batch_data,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length,
            ).to('cuda')

            inputs = {k: v.cuda() for k, v in inputs.items()}

            #TODO: 支持encoder-only T5模型
            if "T5" in type(self.encoder).__name__:
                # T5-based retrieval model
                decoder_input_ids = torch.zeros(
                    (inputs['input_ids'].shape[0], 1), dtype=torch.long
                ).to(inputs['input_ids'].device)
                output = self.encoder(
                    **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
                )
                embeddings = output.last_hidden_state[:, 0, :]

            else:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(output.pooler_output, 
                                    output.last_hidden_state, 
                                    inputs['attention_mask'],
                                    self.pooling_method)
                if  "dpr" not in self.retrieval_method:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            embeddings = cast(torch.Tensor, embeddings)
            embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        """基于BERT等embedding模型构建faiss稠密索引。"""

        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")
        
        self.encoder, self.tokenizer = load_model(model_path = self.model_path, 
                                                  use_fp16 = self.use_fp16)
        if self.embedding_path is not None:
            hidden_size = self.encoder.config.hidden_size
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            all_embeddings = self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus

        # build index
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)
        
        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)

        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")


MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    'jina': 'mean'
}


# 主流程入口，负责参数解析、索引构建。
def main():
    parser = argparse.ArgumentParser(description = "Creating index.")

    # 基本参数
    parser.add_argument('--retrieval_method', type=str, help='检索方法（如e5、bm25等）')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径')
    parser.add_argument('--corpus_path', type=str, help='语料库路径')
    parser.add_argument('--save_dir', default= 'indexes/',type=str, help='索引保存目录')

    # Dense索引参数
    parser.add_argument('--max_length', type=int, default=180, help='最大输入长度')
    parser.add_argument('--batch_size', type=int, default=512, help='批处理大小')
    parser.add_argument('--use_fp16', default=False, action='store_true', help='是否使用FP16')
    parser.add_argument('--pooling_method', type=str, default=None, help='池化方法')
    parser.add_argument('--faiss_type',default=None,type=str, help='faiss索引类型')
    parser.add_argument('--embedding_path', default=None, type=str, help='embedding文件路径')
    parser.add_argument('--save_embedding', action='store_true', default=False, help='是否保存embedding')
    parser.add_argument('--faiss_gpu', default=False, action='store_true', help='是否使用GPU加速faiss')

    args = parser.parse_args()

    # 自动推断池化方法
    if args.pooling_method is None:
        pooling_method = 'mean'
        for k,v in MODEL2POOLING.items():
            if k in args.retrieval_method.lower():
                pooling_method = v
                break
    else:
        if args.pooling_method not in ['mean','cls','pooler']:
            raise NotImplementedError
        else:
            pooling_method = args.pooling_method

    # 构建索引
    index_builder = Index_Builder(
                        retrieval_method = args.retrieval_method,
                        model_path = args.model_path,
                        corpus_path = args.corpus_path,
                        save_dir = args.save_dir,
                        max_length = args.max_length,
                        batch_size = args.batch_size,
                        use_fp16 = args.use_fp16,
                        pooling_method = pooling_method,
                        faiss_type = args.faiss_type,
                        embedding_path = args.embedding_path,
                        save_embedding = args.save_embedding,
                        faiss_gpu = args.faiss_gpu
                    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
