# !/usr/bin/env python
# -*-coding:utf-8 -*-
import random
import os
import torch
import pickle
import torch
import numpy as np
import json
import glob
from vllm import LLM, SamplingParams

from torch import Tensor
from typing import Union, List, Dict, Optional
from datasets import Dataset
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
from retriever import get_retriever_fn

def load_json_dataset(
        data_dir: str,
        file_pattern: str = "*.jsonl",  # 支持 *.jsonl 或 *.json，也可写 *.[jsonl|json] 匹配两种格式
        encoding: str = "utf-8",
        ignore_empty_lines: bool = True,
) -> Union[Dataset, List[dict], Dict[str, Union[Dataset, List[dict]]]]:
    # 1. 基础路径校验
    if not os.path.isdir(data_dir):
        raise ValueError(f"数据文件夹不存在: {data_dir}")

    file_paths = glob.glob(os.path.join(data_dir, file_pattern))
    if not file_paths:
        raise ValueError(f"未找到匹配 '{file_pattern}' 的文件（路径：{data_dir}）")
    file_paths.sort()

    # 2. 读取文件（区分 jsonl 和 json 格式）
    split_data: Dict[str, List[dict]] = {}
    for file_path in file_paths:
        split_name = os.path.splitext(os.path.basename(file_path))[0]
        split_data[split_name] = []
        file_ext = os.path.splitext(file_path)[1].lower()  # 获取文件后缀（小写，兼容 .JSON、.JSONL）

        try:
            with open(file_path, "r", encoding=encoding) as f:
                # 2.1 处理 jsonl 格式（.jsonl 后缀）
                if file_ext == ".jsonl":
                    for line_num, line in enumerate(f, 1):
                        # 处理空行
                        line_stripped = line.strip()
                        if ignore_empty_lines and not line_stripped:
                            continue

                        # 解析单行 JSON
                        try:
                            data = json.loads(line_stripped)
                            # 确保解析结果是字典（兼容单行合法JSON）
                            if isinstance(data, dict):
                                data["split"] = split_name  # 标注数据来源 split
                                split_data[split_name].append(data)
                            else:
                                raise ValueError(f"非字典类型 JSON 数据")
                        except json.JSONDecodeError as e:
                            raise ValueError(f"文件 {file_path} 第 {line_num} 行 JSON 解析失败：{e}")

                # 2.2 处理 json 格式（.json 后缀）
                elif file_ext == ".json":
                    try:
                        # 读取整个文件内容并解析完整 JSON
                        json_content = json.load(f)

                        # 情况1：JSON 是数组（每个元素为字典）
                        if isinstance(json_content, list):
                            for elem in json_content:
                                if isinstance(elem, dict):
                                    elem["split"] = split_name
                                    split_data[split_name].append(elem)
                                else:
                                    raise ValueError(f"JSON 数组中包含非字典元素")

                        # 情况2：JSON 是单个字典
                        elif isinstance(json_content, dict):
                            json_content["split"] = split_name
                            split_data[split_name].append(json_content)

                        # 情况3：不支持的 JSON 类型（如字符串、数字等）
                        else:
                            raise ValueError(f"不支持的 JSON 根类型：{type(json_content).__name__}（仅支持数组或字典）")

                    except json.JSONDecodeError as e:
                        raise ValueError(f"文件 {file_path} 完整 JSON 解析失败：{e}")

                # 2.3 不支持的文件后缀
                else:
                    raise ValueError(f"不支持的文件格式：{file_ext}（仅支持 .json 和 .jsonl）")

        except (PermissionError, OSError) as e:
            raise RuntimeError(f"读取文件 {file_path} 失败：{e}")

    # 3. 生成返回结果（转换为 Dataset）
    result = {}
    for split_name, data_list in split_data.items():
        ds = Dataset.from_list(data_list)
        result[split_name] = ds
    return result

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"可训练参数量: {trainable_params}")
    print(f"总参数量: {all_params}")
    print(f"可训练参数占比: {100 * trainable_params / all_params:.2f}%")

def get_dir(root, sub_dir):
    """
    通过root路径和子目录名称获取完整路径
    :param root:
    :param sub_dir:
    :return:
    """
    path = os.path.join(root, sub_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_dir_data(root, data_name):
    """
    通过root路径和子目录名称获取完整路径
    :param root:
    :param sub_dir:
    :return:
    """
    if data_name in ['AIME2024', 'AIME2025','GPQA', 'GSM8K', "AMC", "MATH500", "HMMT2025",
                     'PLOS', 'eLife', 'Cochrane', 'SumPubmed', 'MedQsum', 'ACI-Bench', 'MTS-Diag']:
        sub_dir = 'datas/' + data_name
    elif data_name in ['DiagnosisArena', 'ReDis', 'CupCase', 'MediQ', 'PubHealth'] + ['QASC', 'LOGIQA', 'ReClor']:
        sub_dir = 'datas/' + data_name
    elif data_name in ['MMLU', 'MMLU-Pro', 'MedBullets', 'MedExQA', 'MedMCQA','MedQA', 'PubMedQA', 'AfrimedQA', 'MedxpertQA-R', 'MedxpertQA-U']:
        sub_dir = 'datas/MedQA-collections/' + data_name.lower()
    else:
        raise NotImplementedError(f"Dataset {data_name} is not supported.")
    path = os.path.join(root, sub_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def set_random_seed(seed):
    """ 设置随机种子以确保代码的可重复性 """
    random.seed(seed)       # Python 内置的随机库
    np.random.seed(seed)    # NumPy 库
    torch.manual_seed(seed) # PyTorch 库

    # 如果您使用 CUDA，则还需要添加以下两行代码
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as f:
        data = pickle.dump(data, f)
    print("File has beeen saved to {}.".format(file_path))
    return




def formulate_dem(retrival_icl, demonstration_tem):
    demonstrations = ''
    for id, (q, a) in enumerate(retrival_icl):
        demonstrations += PromptTemplate.from_template(demonstration_tem).format(
            id=id + 1,
            question=q,
            answer=a,
        )
        demonstrations += '\n'
    return demonstrations


def preprocess_sft_function(example):
    """数据预处理函数"""
    # 我们使用 'text' 字段
    # 你可以根据不同数据集调整字段名
    # 构建一个包含 system, user/prompt, 和 assistant/answer 的对话列表
    messages = [
        {"role": "system", "content": example['system']},
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": example['format_answer']}
    ]
    return {'messages': messages}


def preprocess_grpo_function(example):
    """数据预处理函数"""
    # 我们使用 'text' 字段
    # 你可以根据不同数据集调整字段名
    # 构建一个包含 system, user/prompt, 和 assistant/answer 的对话列表
    messages = [
        {"role": "system", "content": example['system']},
        {"role": "user", "content": example['prompt']},
    ]
    return {'prompt': messages, 'answer': example['format_answer']}

def preprocess_ppo_function(example):
    """数据预处理函数"""
    pass





def get_cot_answer(synthetic_llm, synthetic_tokenizer, sampling_params, cot_tem, question, format_answer):
    prompt = PromptTemplate.from_template(cot_tem)
    user_prompt = prompt.format(question=question,
                                answer=format_answer
                                )  # train不一定要增强
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_prompt}
    ]
    # 3. 调用 chat() 方法进行对话生成
    # 注意：llm.chat() 的 messages 参数需要是 *一个* 对话列表，或者 *多个* 对话列表的列表（用于批量处理）。
    # 如果只处理一个对话，messages 应该是一个列表的列表：[messages]
    # outputs = synthetic_llm.chat(
    #     messages=[messages],  # 将单个对话列表放入一个列表中
    #     sampling_params=sampling_params
    # )
    cot = get_completions(messages, synthetic_llm, synthetic_tokenizer, sampling_params)
    # print("Check user", user_prompt)
    #
    # print("Check cot", cot)


    cot_answer = cot + '\n' + format_answer
    return cot_answer, cot

def get_completions(messages, synthetic_llm, synthetic_tokenizer, sampling_params, model_name=None):
    prompts = synthetic_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # 是否开启思考模式，默认为 True
    )
    outputs = synthetic_llm.generate(prompts, sampling_params)
    text = outputs[0].outputs[0].text
    return text






def set_logging_file(model_path, root, benchmark_name):
    """
    设置日志文件路径
    :param output_root:
    :param log_file_name:
    :return:
    """
    model_name = model_path.split("/")[-1]
    log_dir = root + '/' + f"logs/{benchmark_name}/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{model_name}.txt")

    return log_file


def transfer_dataset_format(data_name, model_name, config):
    # copy到baseline需要的格式
    if model_name in ['AgentSimp', 'MDAgents',  'MedAgents', 'ColaCare']:
        print("You can use baseline in MedAgents and MedAgentBoard, e.g. AgentsBoard & AgentsSimp!")
        print("Compatible data format for this baselines. Please make sure you have the dataset.")
        data_root = get_dir_data(config['ROOT'], data_name)
        train_data_path = os.path.join(data_root, f'train_dataset_tem_noicl_{config["TASK"]}.pkl')
        test_data_path = os.path.join(data_root, f'test_dataset_tem_noicl_{config["TASK"]}.pkl')
        print("TRAIN DATA PATH", train_data_path)
        print("TEST DATA PATH", test_data_path)
    elif model_name in ['TTRL']:
        print("You can use baseline in TTRL, e.g. verl!")
        print("Not compatible data format for TTRL. First Run src/baselines/TTRL/verl/data/preprocess.py")
        data_root = get_dir_data(config['ROOT'], data_name)
        train_data_path = os.path.join(data_root, f'train_dataset_tem_noicl_{config["TASK"]}.pkl')
        test_data_path = os.path.join(data_root, f'test_dataset_tem_noicl_{config["TASK"]}.pkl')
        train_dataset = load_pickle(train_data_path)
        test_dataset = load_pickle(test_data_path)
        from src.baselines.TTRL.verl.data.preprocess import ttrl_data_format
        save_path = ttrl_data_format(train_dataset, test_dataset, data_name)
        print("DATA PATH ", data_name + '-TTT ', f"has been generated in {save_path}!")
        print("Note change conda env to ttrl to run the baseline.")



##########embedding model
def get_embedding_model(model_name, config, device: Optional[str] = 'cuda'):
    hub_root = config['HUB_ROOT']
    emb_path = os.path.join(hub_root, model_name)
    print("Embedding model path:", emb_path, device)

    if model_name == 'MedCPT':
        embedding_model = CustomMedCPTEmbeddings(
            model_name=emb_path,
            max_length=64,
            device=device,
        )
    elif model_name == 'E5':
        # 这里是E5-V2
        model_kwargs = {
            "device": device,  # 把device参数移到这里
        }
        embedding_model = HuggingFaceEmbeddings(
            model_name=emb_path,
            model_kwargs=model_kwargs
        )
    elif model_name == 'BGE':
        embedding_model = CustomBGELargeEmbeddings(
            model_name=emb_path,
            # 可选：添加s2p任务指令前缀，比如
            instruction="为这个句子生成用于检索的表示：",
            max_length=512
        )
    elif model_name == 'BMRetriever': # yong 1B
        embedding_model = CustomBMRetrieverEmbeddings(
        model_name=emb_path,
        task_description='Given a query, retrieve similar instance',
        max_length=64
    )
    else:
        raise NotImplementedError(f"Embedding model {model_name} is not supported.")
    return embedding_model







class CustomMedCPTEmbeddings(Embeddings):
    """自定义MedCPT Embeddings类，兼容LangChain接口"""

    def __init__(
            self,
            model_name: str = "/nfs/scratch/czhaobo/huggingface/hub/MedCPT",
            max_length: int = 64,
            device: Optional[str] = None,
    ):
        # 自动选择设备（CPU/GPU）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        # 加载模型和tokenizer（和原代码一致）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _get_embedding(self, text: str) -> List[float]:
        """生成单条文本的嵌入向量（核心逻辑）"""
        # 手动tokenize，传递正确的参数（和原代码一致）
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",  # 补齐到max_length
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)

        # 生成嵌入，取[CLS] token的输出（和原代码逻辑一致）
        with torch.no_grad():
            outputs = self.model(**encoded)
            embed = outputs.last_hidden_state[:, 0, :].squeeze(0)  # 去掉batch维度

        # 转换为列表返回（符合LangChain Embeddings接口要求）
        return embed.cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本嵌入（LangChain标准接口）"""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """生成单条查询文本的嵌入（LangChain标准接口）"""
        return self._get_embedding(text)





def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """原代码的last_token_pool池化函数，完全保留"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    return embedding


def get_detailed_instruct_query(task_description: str, query: str) -> str:
    """生成带指令的查询文本（原代码逻辑）"""
    return f'{task_description}\nQuery: {query}'


def get_detailed_instruct_passage(passage: str) -> str:
    """生成带指令的文档文本（原代码逻辑）"""
    return f'Represent this passage\npassage: {passage}'


# 2. 自定义LangChain兼容的BMRetriever Embeddings类
class CustomBMRetrieverEmbeddings(Embeddings):
    """自定义BMRetriever Embeddings类，兼容LangChain接口"""

    def __init__(
            self,
            model_name: str = "BMRetriever/BMRetriever-410M",
            task_description: str = 'Given a scientific claim, retrieve documents that support or refute the claim',
            max_length: int = 512,
            device: Optional[str] = None,
    ):
        # 自动选择设备（优先GPU）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.task_description = task_description

        # 加载模型和tokenizer（和原代码一致）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 设置为评估模式

    def _process_texts(self, texts: List[str], is_query: bool = False) -> List[str]:
        """为文本添加指令前缀（区分query/passage）"""
        if is_query:
            return [get_detailed_instruct_query(self.task_description, text) for text in texts]
        else:
            return [get_detailed_instruct_passage(text) for text in texts]
  
    def _generate_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """核心嵌入生成逻辑（支持批量分块处理，修复Tensor和列表相加的错误）"""
        # 配置分块批次大小（关键参数：根据GPU显存调整，建议8/16/32，显存紧张设更小）
        BATCH_SIZE = 8  # 可抽为类参数，如self.embed_batch_size，方便外部配置
        all_embeddings = []  # 存储所有分块的嵌入结果

        # 步骤1：添加指令前缀（整批预处理，不影响显存）
        processed_texts = self._process_texts(texts, is_query=is_query)

        # 核心改造：按BATCH_SIZE分块遍历，逐块生成嵌入
        for i in range(0, len(processed_texts), BATCH_SIZE):
            # 截取当前批次的文本
            batch_texts = processed_texts[i:i + BATCH_SIZE]
            batch_size_current = len(batch_texts)  # 处理最后一批不足BATCH_SIZE的情况

            # 步骤2：Tokenize（保留原逻辑：max_length-1、padding、truncation）
            batch_dict = self.tokenizer(
                batch_texts,
                max_length=self.max_length - 1,  # 预留位置给EOS token
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            # 步骤3：添加EOS token（保留原Tensor拼接逻辑，适配当前小批次）
            # 生成当前批次的EOS token Tensor（形状：[batch_size_current, 1]）
            eos_tokens = torch.tensor([[self.tokenizer.eos_token_id]] * batch_size_current).to(self.device)
            batch_dict['input_ids'] = torch.cat([batch_dict['input_ids'], eos_tokens], dim=1)

            # 同步更新attention_mask（EOS token对应的mask设为1）
            eos_mask = torch.ones((batch_size_current, 1), dtype=torch.long).to(self.device)
            batch_dict['attention_mask'] = torch.cat([batch_dict['attention_mask'], eos_mask], dim=1)

            # 步骤4：重新pad（确保当前批次内长度统一，保留原逻辑）
            batch_dict = self.tokenizer.pad(
                batch_dict,
                padding=True,
                return_attention_mask=True,
                return_tensors='pt'
            ).to(self.device)

            # 步骤5：模型推理生成嵌入（保留no_grad，禁用梯度减少显存占用）
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                # 步骤6：last_token_pool池化（保留原核心逻辑）
                batch_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # 将当前批次嵌入移到CPU，避免GPU显存堆积，添加到总结果
            all_embeddings.append(batch_embeddings.cpu())

            # 可选：及时删除当前批次的GPU张量，释放显存（显存极度紧张时启用）
            del batch_dict, outputs, batch_embeddings
            torch.cuda.empty_cache()

        # 拼接所有分块的嵌入结果，转换为LangChain要求的List[List[float]]格式
        final_embeddings = torch.cat(all_embeddings, dim=0).numpy().tolist()
        return final_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入（LangChain标准接口）"""
        return self._generate_embeddings(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        """生成单条查询嵌入（LangChain标准接口）"""
        return self._generate_embeddings([text], is_query=True)[0]




class CustomBGELargeEmbeddings(Embeddings):
    """自定义BGE-large-zh-v1.5 Embeddings类，兼容LangChain接口"""

    def __init__(
            self,
            model_name: str = "BAAI/bge-large-zh-v1.5",
            instruction: Optional[str] = None,  # s2p任务的查询指令前缀
            max_length: int = 512,  # BGE默认最大长度，可根据需求调整
            device: Optional[str] = None,
    ):
        # 自动选择设备（优先GPU）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.instruction = instruction
        self.max_length = max_length

        # 加载模型和tokenizer（和原代码一致）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 保持评估模式，和原代码一致

    def _generate_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """核心嵌入生成逻辑（复刻原代码所有步骤）"""
        # 步骤1：处理指令前缀（仅query添加，适配s2p任务）
        processed_texts = texts
        if is_query and self.instruction is not None:
            processed_texts = [self.instruction + text for text in texts]

        # 步骤2：Tokenize（和原代码一致：padding、truncation）
        encoded_input = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)

        # 步骤3：模型推理生成嵌入
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # 步骤4：CLS pooling（原代码核心逻辑：取[:, 0]）
            sentence_embeddings = model_output[0][:, 0]

        # 步骤5：L2归一化（原代码关键步骤）
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        # 转换为列表（符合LangChain接口要求）
        return sentence_embeddings.cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档嵌入（LangChain标准接口，不加指令前缀）"""
        return self._generate_embeddings(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        """生成单条查询嵌入（LangChain标准接口，可选加指令前缀）"""
        return self._generate_embeddings([text], is_query=True)[0]


def re_gather_dataset(train_dic, test_dic, index_root, special_op, config, kg=None):
    """
    :param train_dataset: 用作Index
    :param test_dataset: 用于遍历
    :param special_op:
    :param kg:
    :return:
    """
    without_train = ["AIME2024", "AIME2025", 'MATH500', 'GPQA', 'HMMT2025', 'CupCase']
    if config['DATASET'] in without_train:
        data_mode = 'without_train'
        exit("目前这些数据集不支持检索增强，请使用有train的版本！")
    else:
        data_mode = 'with_train'

    # embedding model
    print("Initializing embedding model...", config['EMB'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = get_embedding_model(config['EMB'], config, device)
    # 改造
    if special_op == 'ICL': # retrieval top-k neighbors
        index_dataset = Dataset.from_dict({
            'text': [item['Q'] for item in train_dic], # 这里只使用题干相似性， 也可以使用anwer进行aug
            'answer': [item['A'] for item in train_dic],
            'id': list(range(len(train_dic)))
        })
        print('Emb sample', index_dataset[0])
        
        retriever = get_retriever_fn(index_dataset, embedding_model, text_column='text',
                                     index_path=index_root, retrieval_type=config['RETRIEVAL_TYPE'])
        topk = config['TOPK']
        for query_index in range(len(test_dic)):
            if data_mode == 'without_train':  # 在train上做的, 重新实现, 取出本身, 其实检索K + 1 就行 [(id, score)]
                similar_indices = retriever.topk_retrieval(query_index, topk) # [id]
            elif data_mode == 'with_train': # 在test上做的, 这个很方便，要么使用unicorn，要么使用openicl
                similar_indices = retriever.query_by_text(test_dic[query_index]['Q'], topk) # 这种简单，直接retrieval, [1,2,3,4]
            test_dic[query_index]['similar_indices'] = similar_indices
            test_dic[query_index]['retrieved_texts'] = [(index_dataset[i]['text'], index_dataset[i]['answer']) for i in similar_indices]
            test_dic[query_index]['similar_indices_kg'] = []
            test_dic[query_index]['retrieved_texts_kg'] = []
    elif special_op == 'RAG': # retrieval top-k kg， 使用unicorn，然后再进行kg的retrieval
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


def syn_context_questions(test_file_check, template_config, config):
    # 读取数据
    test_dataset = load_pickle(test_file_check)
    gen_prompt = template_config['public']['generate_questions']

    synthetic_llm = LLM(model=config['LLM_PATH'],
                        trust_remote_code=True,
                        gpu_memory_utilization=0.7,
                        tensor_parallel_size=2)
    synthetic_tokenizer = AutoTokenizer.from_pretrained(config['LLM_PATH'])

    # think mode params
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=config['MAX_LEN'],
        min_p=0,
        # repetition_penalty=1.2,
        # presence_penalty=1.5,
        top_p=0.95,  # 增加生成稳定性
        top_k=20,
    )

    synthetic_qa_pair = []
    for i in range(len(test_dataset)):
        retrieval_icl = test_dataset[i].get('retrieved_texts', [])
        retrieval_icl = retrieval_icl[:config['TOPK']]
        for j in range(len(retrieval_icl)):
            question = retrieval_icl[j][0]
            user_prompt = gen_prompt.format(text=question)
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": user_prompt}
            ]

            # ========== 新增：最多重试2次生成逻辑 ==========
            max_retry = 2  # 最大重试次数
            retry_count = 0
            parse_success = False

            while retry_count <= max_retry and not parse_success:
                # 调用生成函数
                generated_lis = get_completions(messages, synthetic_llm, synthetic_tokenizer, sampling_params)

                # 过滤空结果
                if not generated_lis or generated_lis.strip() == "":
                    retry_count += 1
                    print(f"第{i}条数据第{j}个检索文本生成结果为空，重试第{retry_count}次")
                    continue

                try:
                    # 清理格式并解析
                    clean_text = generated_lis.strip().replace('，', ',').replace('（', '(').replace('）', ')')
                    import ast
                    parsed_qa_list = ast.literal_eval(clean_text)

                    # 验证解析结果格式
                    if isinstance(parsed_qa_list, list):
                        for qa_item in parsed_qa_list:
                            if isinstance(qa_item, (tuple, list)) and len(qa_item) == 2:
                                gen_q = qa_item[0].strip()
                                gen_a = qa_item[1].strip()
                                if gen_q and gen_a:
                                    synthetic_qa_pair.append({
                                        "source_question": question,
                                        "generated_question": gen_q,
                                        "generated_answer": gen_a
                                    })
                                    test_dataset[i]['context_qa_pair'] = synthetic_qa_pair
                        parse_success = True  # 解析成功，终止重试
                        if retry_count > 0:
                            print(f"第{i}条数据第{j}个检索文本重试{retry_count}次后解析成功")
                    else:
                        raise ValueError("生成结果不是列表格式")

                except (SyntaxError, ValueError, NameError) as e:
                    retry_count += 1
                    if retry_count <= max_retry:
                        print(f"第{i}条数据第{j}个检索文本解析失败（{e}），重试第{retry_count}次")
                    else:
                        print(f"第{i}条数据第{j}个检索文本重试{max_retry}次仍失败，跳过该条")

            # 可选：打印进度信息
            if (i * len(retrieval_icl) + j + 1) % 10 == 0:
                print(
                    f"已处理 {i * len(retrieval_icl) + j + 1} 条检索文本，成功生成 {len(synthetic_qa_pair)} 条有效问答对")

    # 存储回文件, 在test_file_check的.pkl前加上-context
    output_path = test_file_check.replace('.pkl', '-context.pkl')
    save_pickle(test_dataset, output_path)
    print("Have created new test dataset with contextual QA pairs at:", output_path)

    # 返回最终生成的问答对列表
    return synthetic_qa_pair


def disturb_context(test_file_check, num_k=3):
    test_dataset = load_pickle(test_file_check)
    for i in range(len(test_dataset)):
        context_qa_pair = test_dataset[i].get('context_qa_pair', [])
        # 打乱顺序
        for _k in range(num_k):
            random.shuffle(context_qa_pair)
            test_dataset[i][f'context_qa_pair_disturbed_{_k}'] = context_qa_pair.copy()
    output_path = test_file_check.replace('.pkl', f'-disturbed{num_k}.pkl')
    save_pickle(test_dataset, output_path)
    print("Have created new test dataset with disturbed contextual QA pairs at:", output_path)

