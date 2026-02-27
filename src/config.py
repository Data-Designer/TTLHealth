# !/usr/bin/env python
# -*-coding:utf-8 -*-
from langchain_core.runnables.config import RunnableConfig

MedQA_PARAMS = {
    'SEED': 42,
    'USE_CUDA': True,
    'GPU': '1',
    'GET_COT': True,

    # generation
    'TEMP': 0.6,
    'DO_SAMPLE': True,
    'MAX_LEN': 8192,  # 32628
    'MAX_RETRY': 20,
    'AVG': 1,
    'ENTROPY_THRE': 3.0,
    'LONG_ENTROPY_THRE': 3.0,

    # TTL optimization
    'ITER_NUM': 3,
    'LR': 0.01,
    'ENTROPY_WEIGHT': 0.05,
    'STRU_WEIGHT': 0.05,
    'QUERY_WEIGHT': 2,
    'ENTROPY_WIN': 4,
    'ENTROPY_TOKEN_MIN': 25,
    'LONG_ENTROPY_WIN': 4,
    'LONG_ENTROPY_TOKEN_MIN': 25,
    'MIN_STD': 0.5,
    'MIN_THRESHOLD': 1.8,
    'PRETRAIN': False,
    # SFT params
    'PRETRAIN_OP': 'SFT',
    'PRETRAIN_LORA_RANK': 8,
    'PRETRAIN_LORA_ALPHA': 16,
    'PRETRAIN_LORA_DROP': 0.1,
    'PRETRAIN_EPOCHS': 2,
    'PRETRAIN_BATCH_SIZE': 4,
    'GRADIENT_BACK': 4,
    'PRETRAIN_LR': 1e-4,
    'PRETRAIN_WD': 0,
    'PRETRAIN_GEN_NUM': 4,
    'MAX_SFT_LEN': 4096,
    'MAX_GEN_LEN': 4096
}

class UNIFYCONFIG():
    """
    config class
    """
    # basic info
    MODEL = 'ours'
    TASK = 'FREE' # MCQ, FREE
    DATASET = 'MedQA'
    OP = 'ICL'

    # PATH
    ROOT = '/nfs/usrhome2/czhaobo/TTLHealth/'

    EMB = 'E5'# 'E5'#'BMRetriever'# 'MedCPT'#'BMRetriever' # 'E5' # 一般不要换，不然得重新构建embedding。
    HUB_ROOT = "/nfs/scratch/czhaobo/huggingface/hub/"
    RETRIEVAL_TYPE = 'topk' # topk, ann
    TOPK = 3

    LLM = 'icl'
    LLM_PATH = "/nfs/scratch/czhaobo/huggingface/hub/qwen25-7B-instruct" # qwen25-7B-instruct

    TTL_STATE = 'online' # online, offline
    # model info

    KG_PATH = "/home/czhaobo/RAGHealth/data/raw/primekg/"
    TEXT_PATH = "/home/czhaobo/RAGHealth/data/ready/pubmed_results_node_20250314_112754.json"


    # train dataset info
    DATASET_PARAMS = {
        'MedQA': MedQA_PARAMS,
    }

    # 对baseline
    # DO_SAMPLE = False
    MASK_SPECIAL = True
    PARALLEL = False 
    DEVICE = 'auto' # auto, cuda, cpu
    MAX_PARALLEL_GPUS = 4

    @classmethod
    def get_params(cls):
        return cls.DATASET_PARAMS.get(cls.DATASET, {})



unify_config = {**vars(UNIFYCONFIG), **UNIFYCONFIG.get_params()}
unify_config = {k: v for k, v in unify_config.items() if not k.startswith('__')}
