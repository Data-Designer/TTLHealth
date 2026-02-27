# !/usr/bin/env python
# -*-coding:utf-8 -*-

import numpy as np
import faiss
from datasets import Dataset
from langchain.embeddings.base import Embeddings
import pickle
import os




def get_retriever_fn(dataset, embedding_model, text_column='text', index_path=None, retrieval_type='topk'):
    if retrieval_type == 'topk':
        return TopKRetrievalWithIndex(dataset, embedding_model, text_column, index_path)
    else:
        raise ValueError(f"不支持的检索类型: {retrieval_type}")



class TopKRetrievalWithIndex:
    """
    使用预计算嵌入向量和Faiss索引的Top-K检索器
    """

    def __init__(self, dataset: Dataset, embedding_model: Embeddings,
                 text_column: str = 'text', index_path: str = None):
        """
        初始化检索器

        Args:
            dataset: 数据集
            embedding_model: LangChain嵌入模型
            text_column: 文本列名
            index_path: 索引保存路径，如果提供则尝试加载已有索引
        """
        self.dataset = dataset
        self.embedding_model = embedding_model
        self.text_column = text_column
        self.index_path = index_path

        # 检查是否已有保存的索引
        if index_path and os.path.exists(index_path):
            self._load_index()
        else:
            self._build_index()

    def _build_index(self):
        """构建Faiss索引"""
        print("正在生成文本嵌入向量...")
        texts = self.dataset[self.text_column]
        self.embeddings = self.embedding_model.embed_documents(texts)
        self.embeddings = np.array(self.embeddings, dtype=np.float32)

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        faiss.normalize_L2(self.embeddings)

        self.index.add(self.embeddings)

        print(f"索引构建完成，共 {len(self.embeddings)} 个向量")

        # 保存索引
        if self.index_path:
            self._save_index()

    def _save_index(self):
        """保存索引到文件"""
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else '.', exist_ok=True)

        # 保存Faiss索引
        faiss.write_index(self.index, f"{self.index_path}.faiss")

        # 保存嵌入向量和其他元数据
        with open(f"{self.index_path}.pkl", 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'text_column': self.text_column
            }, f)

        print(f"索引已保存到 {self.index_path}")

    def _load_index(self):
        """从文件加载索引"""
        try:
            # 加载Faiss索引
            self.index = faiss.read_index(f"{self.index_path}.faiss")

            # 加载嵌入向量和元数据
            with open(f"{self.index_path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.text_column = data.get('text_column', 'text')

            print(f"索引已从 {self.index_path} 加载，共 {len(self.embeddings)} 个向量")
        except Exception as e:
            print(f"加载索引失败: {e}，将重新构建索引")
            self._build_index()

    def topk_retrieval(self, query_idx: int, k: int, return_scores: bool = False):
        """
        执行Top-K检索

        Args:
            query_idx: 查询样本索引
            k: 返回的最相似样本数量
            return_scores: 是否返回相似度分数

        Returns:
            检索结果
        """
        if k <= 0:
            raise ValueError("k必须大于0")
        if query_idx < 0 or query_idx >= len(self.dataset):
            raise ValueError("query_idx超出数据集范围")

        query_embedding = self.embeddings[query_idx:query_idx + 1].copy()
        faiss.normalize_L2(query_embedding)

        actual_k = min(k + 1, len(self.dataset))
        scores, indices = self.index.search(query_embedding, actual_k)

        result_indices = []
        result_scores = []

        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != query_idx and len(result_indices) < k:
                result_indices.append(idx)
                result_scores.append(score)

        if return_scores:
            return list(zip(result_indices, result_scores))
        else:
            return result_indices


    def query_by_text(self, query_text: str, k: int, return_scores: bool = False):
        """
        通过文本直接查询（不需要文本在数据集中）

        Args:
            query_text: 查询文本
            k: 返回的最相似样本数量
            return_scores: 是否返回相似度分数

        Returns:
            检索结果
        """
        if k <= 0:
            raise ValueError("k必须大于0")

        # 生成查询文本的嵌入向量
        query_embedding = self.embedding_model.embed_documents([query_text])
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # 标准化
        faiss.normalize_L2(query_embedding)

        # 执行搜索
        scores, indices = self.index.search(query_embedding, k)

        if return_scores:
            return list(zip(indices[0], scores[0]))
        else:
            return indices[0].tolist()
