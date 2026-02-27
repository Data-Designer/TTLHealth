# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel

from retriever import get_retriever_fn
from langchain_huggingface import HuggingFaceEmbeddings

from utils import load_json_dataset, get_embedding_model
from preprocess import preprocess_med_collections, preprocess_sum_collections, preprocess_diag_collections, \
    preprocess_math_collections


def get_task_fn(config):
    """
    :return: task_mode, task_fn, mode_desc ('MCQ', 'Generation', 'MATCH')
    """
    if config['DATASET'] in ['AIME2024', 'AIME2025']:
        pretrain, task_fn, task_mode = config['PRETRAIN'], math_task_fn, config['TASK']
    elif config['DATASET'] in ['GPQA']:
        pretrain, task_fn, task_mode = config['PRETRAIN'], mcq_task_fn, config['TASK']
    elif config['DATASET'] in ['GSM8K']:
        pretrain, task_fn, task_mode = config['PRETRAIN'], math_task_fn, config['TASK']
    elif config['DATASET'] in ['HMMT2025', 'AMC', 'MATH500']:
        pretrain, task_fn, task_mode = config['PRETRAIN'], math_task_fn, config['TASK']
    elif config['DATASET'] in ['MMLU', 'MMLU-Pro', 'MedBullets', 'MedExQA', 'MedMCQA', 'MedQA', 'PubMedQA', 'AfrimedQA',
                               'MedxpertQA-R', 'MedxpertQA-U']:
        pretrain, task_fn, task_mode = config['PRETRAIN'], mcq_task_fn, config['TASK']
    elif config['DATASET'] in ['PLOS', 'eLife', 'Cochrane', 'SumPubmed', 'MedQsum', 'ACI-Bench', 'MTS-Diag']:
        pretrain, task_fn, task_mode = config['PRETRAIN'], sum_task_fn, config['TASK']
    elif config['DATASET'] in ['DiagnosisArena', 'ReDis', 'CupCase', 'MediQ', 'PubHealth'] + ['QASC', 'LOGIQA', 'ReClor']:
        pretrain, task_fn, task_mode = config['PRETRAIN'], diag_task_fn, config['TASK']
    else:
        raise NotImplementedError(f"Dataset {config['DATASET']} is not supported.")
    return pretrain, task_fn, task_mode


def get_dataset_fn(data_name, data_root):
    """
    :param data_name:
    :param task_fn:
    :return: datasets
    """

    if data_name == 'AIME2024':
        dataset = load_dataset("HuggingFaceH4/aime_2024", cache_dir=data_root)
    elif data_name == 'AIME2025':
        dataset = load_dataset("yentinglin/aime_2025", cache_dir=data_root)
    elif data_name == 'GPQA':
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", cache_dir=data_root)
    elif data_name == 'LOGIQA':
        dataset = load_dataset("lucasmccabe/logiqa", cache_dir=data_root, trust_remote_code=True)
    elif data_name == 'QASC':
        dataset = load_dataset("allenai/qasc", cache_dir=data_root)
    elif data_name == 'ReClor':
        dataset = load_dataset("metaeval/reclor", cache_dir=data_root)
    elif data_name == 'GSM8K':
        dataset = load_dataset("openai/gsm8k", "main", cache_dir=data_root)
    elif data_name == 'HMMT2025':
        dataset = load_dataset("MathArena/hmmt_feb_2025", cache_dir=data_root)
    elif data_name == "AMC":
        dataset = load_dataset("AI-MO/NuminaMath-CoT", cache_dir=data_root)
    elif data_name == 'MATH500':
        dataset = load_dataset("HuggingFaceH4/MATH-500", cache_dir=data_root)
    elif data_name in ['MMLU', 'MMLU-Pro', 'MedBullets', 'MedExQA', 'MedMCQA', 'MedQA', 'PubMedQA', 'AfrimedQA',
                       'MedxpertQA-R', 'MedxpertQA-U']:
        print("Check MedQA-collections dir")  # 这里仿照medagents benchmark其处理即可。
        cache_dir = data_root
        dataset = load_json_dataset(cache_dir)
    elif data_name == 'PLOS':
        cache_dir = data_root
        dataset = load_json_dataset(cache_dir, file_pattern="*.json")
    elif data_name == 'eLife':
        cache_dir = data_root
        dataset = load_json_dataset(cache_dir, file_pattern="*.json")
    elif data_name == 'Cochrane':
        cache_dir = data_root
        dataset = load_json_dataset(cache_dir, file_pattern="*.json")
    elif data_name == 'SumPubmed':
        dataset = load_dataset("Blaise-g/SumPubmed", cache_dir=data_root)
    elif data_name == 'MedQsum':
        print("注意dataset版本,MedQsum需要用低版本datasetS")
        dataset = load_dataset("lighteval/me_q_sum", cache_dir=data_root)
    elif data_name == 'ACI-Bench':
        dataset = load_dataset("ClinicianFOCUS/ACI-Bench-Refined", cache_dir=data_root)
    elif data_name == 'MTS-Diag':
        dataset = load_dataset("beanham/medsum", cache_dir=data_root)
    elif data_name == 'DiagnosisArena':
        dataset = load_dataset("shzyk/DiagnosisArena", cache_dir=data_root)
    elif data_name == 'ReDis':
        dataset = load_dataset("guan-wang/ReDis-QA", cache_dir=data_root)
    elif data_name == 'CupCase':
        dataset = load_dataset("ofir408/CupCase", cache_dir=data_root)
    elif data_name == 'MediQ':
        dataset = load_dataset("stellalisy/mediQ", cache_dir=data_root)
    elif data_name == 'PubHealth':
        dataset = load_dataset("Joshua-Harris/PubHealthBench", cache_dir=data_root)
    else:
        raise NotImplementedError(f"Dataset {data_name} is not supported.")
    return dataset


def math_task_fn(dataset, dataset_name, task_name='FREE', sample_num=None):
    """
    :param dataset:
    :param task_mode:
    :return:
    """
    if dataset_name in ['AIME2024', 'AIME2025', 'GSM8K', 'HMMT2025', 'AMC', 'MATH500']:
        QAs = preprocess_math_collections(dataset, dataset_name)
        if task_name == 'FREE':
            QAs = QAs
        else:
            raise NotImplementedError(f"Task {task_name} is not supported for dataset {dataset_name}.")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    if sample_num is not None and len(QAs) > sample_num:
        QAs = QAs[:sample_num]
    print("Number of samples:", len(QAs))

    return QAs


def sum_task_fn(dataset, dataset_name, task_name='FREE', sample_num=None):
    """
    :param dataset:
    :param task_mode:
    :return:
    """
    # 数据提取
    if dataset_name in ['PLOS', 'eLife', 'Cochrane', 'SumPubmed', 'MedQsum', 'ACI-Bench', 'MTS-Diag']:
        dataset = preprocess_sum_collections(dataset, dataset_name)
        if task_name == 'FREE':
            QAs = [{'Q': x, 'A': y}
                   for x, y in zip(dataset['article'], dataset['summary'])]
        else:
            raise NotImplementedError(f"Task {task_name} is not supported for dataset {dataset_name}.")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    if sample_num is not None and len(QAs) > sample_num:
        QAs = QAs[:sample_num]
    print("Number of samples:", len(QAs))
    return QAs


def mcq_task_fn(dataset, dataset_name, task_name='MCQ', sample_num=None):
    """
    :param dataset:
    :param task_mode:
    :return:
    """
    # 数据提取
    if dataset_name in ['MMLU', 'MMLU-Pro', 'MedBullets', 'MedExQA', 'MedMCQA', 'MedQA', 'PubMedQA', 'AfrimedQA',
                        'MedxpertQA-R', 'MedxpertQA-U'] + ['GPQA']:
        dataset = preprocess_med_collections(dataset, dataset_name)
        if task_name == 'MCQ':
            QAs = [{'Q': x, 'O': o, 'A': y}
                   for x, o, y in zip(dataset['question'], dataset['options'], dataset['answer'])]
        else:
            raise NotImplementedError(f"Task {task_name} is not supported for dataset {dataset_name}.")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    if sample_num is not None and len(QAs) > sample_num:
        QAs = QAs[:sample_num]
    print("Number of samples:", len(QAs))
    return QAs


def diag_task_fn(dataset, dataset_name, task_name='MCQ', sample_num=None):
    """
    :param dataset:
    :param task_mode:
    :return:
    """
    # 数据提取
    if dataset_name in ['DiagnosisArena', 'ReDis', 'CupCase', 'MediQ', 'PubHealth'] + ['QASC', 'LOGIQA', 'ReClor']:
        dataset = preprocess_diag_collections(dataset, dataset_name)
        if task_name == 'MCQ':
            QAs = [{'Q': x, 'O': o, 'A': y}
                   for x, o, y in zip(dataset['question'], dataset['options'], dataset['answer'])]
        elif task_name == 'FREE':
            QAs = [{'Q': x, 'O': o, 'A': y}
                   for x, o, y in zip(dataset['question'], dataset['options'], dataset['answer_text'])]
        else:
            raise NotImplementedError(f"Task {task_name} is not supported for dataset {dataset_name}.")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    if sample_num is not None and len(QAs) > sample_num:
        QAs = QAs[:sample_num]
    print("Number of samples:", len(QAs))
    return QAs


def split_dataset(dataset, dataset_name):
    if dataset_name in ["AIME2024", "AIME2025", 'GPQA'] + ['HMMT2025']:
        train_dataset, test_dataset = dataset['train'], dataset['train']
        data_mode = 'without_train'
        print(
            "***********Note that this dataset only has train set! Using train as test set! No need Pretrain ! 这些都不能用！***********")
    elif dataset_name in ['MATH500'] + ['DiagnosisArena', 'ReDis', 'CupCase']:
        train_dataset, test_dataset = dataset['test'], dataset['test']
        data_mode = 'without_train'
        print("***********Note that this dataset only has test set! No need Pretrain ! 这些都不能用！***********")
        if dataset_name in ['DiagnosisArena', 'ReDis']:
            print("***********Change the split***********")
            data_mode = 'with_train'
            if dataset_name == 'DiagnosisArena':
                class_label = ClassLabel(names=['A', 'B', 'C', 'D'])
                test_dataset = test_dataset.cast_column("Right Option", class_label)
                print(test_dataset[:2])
                # 分层切分：按label列分层，测试集占10%
                split_dataset = test_dataset.train_test_split(
                    test_size=0.1,
                    seed=42,
                    stratify_by_column="Right Option"
                )
            elif dataset_name == 'ReDis':
                print(test_dataset[:2])
                class_label = ClassLabel(names=[0, 1, 2, 3])
                test_dataset = test_dataset.cast_column("cop", class_label)
                # test_dataset = test_dataset.class_encode_column("cop")
                print(test_dataset[:2])

                split_dataset = test_dataset.train_test_split(
                    test_size=0.1,
                    seed=42,
                    stratify_by_column="cop"
                )
            train_dataset = split_dataset["train"]
            test_dataset = split_dataset["test"]

    elif (dataset_name in ['AMC'] + ['MMLU', 'MMLU-Pro', 'MedBullets', 'MedExQA', 'MedMCQA', 'MedQA', 'PubMedQA',
                                     'AfrimedQA', 'MedxpertQA-R', 'MedxpertQA-U']
          + ['SumPubmed', 'eLife', 'Cochrane', 'PLOS', 'ACI-Bench', 'MTS-Diag', 'MedQsum'] + ['GSM8K'] + ['QASC',
                                                                                                          'LOGIQA','ReClor']):
        if dataset_name in ['ReClor','MedQsum', 'QASC']:
            train_dataset, test_dataset = dataset['train'], dataset['validation']
        else:
            train_dataset, test_dataset = dataset['train'], dataset['test']
        data_mode = 'with_train'
    elif dataset_name in ['MediQ', 'PubHealth']:
        train_dataset, test_dataset = dataset['validation'], dataset['test']
        if dataset_name in ['PubHealth']:
            test_dataset = dataset['reviewed']
        data_mode = 'with_train'

    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported for split dataset.")
    return train_dataset, test_dataset, data_mode


def gather_dataset(dataset_name, data_mode, task_fn, train_dataset, test_dataset, index_root, special_op, config,
                   sample_num=None, kg=None):
    """
    :param train_dataset: 用作Index
    :param test_dataset: 用于遍历
    :param special_op:
    :param kg:
    :return:
    """
    # embedding model
    print("Initializing embedding model...", config['EMB'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = get_embedding_model(config['EMB'], config, device)
    train_dic = task_fn(train_dataset, dataset_name, config['TASK'], sample_num)
    test_dic = task_fn(test_dataset, dataset_name, config['TASK'], sample_num)


    if special_op == 'ICL':
        index_dataset = Dataset.from_dict({
            'text': [item['Q'] for item in train_dic],
            'answer': [item['A'] for item in train_dic],
            'id': list(range(len(train_dic)))
        })
        print('Emb sample', index_dataset[0])
        retriever = get_retriever_fn(index_dataset, embedding_model, text_column='text',
                                     index_path=index_root, retrieval_type=config['RETRIEVAL_TYPE'])
        topk = config['TOPK']
        for query_index in range(len(test_dic)):
            if data_mode == 'without_train':
                similar_indices = retriever.topk_retrieval(query_index, topk)  # [id]
            elif data_mode == 'with_train':
                similar_indices = retriever.query_by_text(test_dic[query_index]['Q'],
                                                          topk)
            test_dic[query_index]['similar_indices'] = similar_indices
            test_dic[query_index]['retrieved_texts'] = [(index_dataset[int(i)]['text'], index_dataset[int(i)]['answer'])
                                                        for i in similar_indices]
            test_dic[query_index]['similar_indices_kg'] = []
            test_dic[query_index]['retrieved_texts_kg'] = []
    elif special_op == 'RAG':
        external_kg = kg
        index_dataset = None
        for query_index in range(len(test_dic)):
            test_dic[query_index]['similar_indices'] = []
            test_dic[query_index]['retrieved_texts'] = []
            test_dic[query_index]['similar_indices_kg'] = ''
            test_dic[query_index]['retrieved_texts_kg'] = ''
        pass
    else:
        for query_index in range(len(test_dic)):
            test_dic[query_index]['similar_indices'] = []
            test_dic[query_index]['retrieved_texts'] = []
            test_dic[query_index]['similar_indices_kg'] = []
            test_dic[query_index]['retrieved_texts_kg'] = []

    return train_dic, test_dic



