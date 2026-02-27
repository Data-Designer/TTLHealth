# !/usr/bin/env python
# -*-coding:utf-8 -*-
import re
from typing import List, Dict, Any, Callable, Tuple


def parse_answer_tag(completion: str) -> str:
    """从 <answer> 标签中提取最终答案。"""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)
    if match:
        answer = match.group(1).strip().replace('\n', ' ')

        if re.match(r'^-?\d+(\.\d+)?$', answer):  # 如果答案是纯数字
            return answer

        return answer
    return ""


def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    奖励模型是否遵循特定的 <think>...<answer> 格式。
    - 奖励值: +1.0 (正确格式) / -0.5 (错误格式)
    """
    rewards = []

    # 定义必须出现的标签模式
    pattern = re.compile(r'<think>.*?<\think>.*?<answer>.*?</answer>', re.DOTALL)

    for completion in completions:
        # 必须包含 <think> 和 <answer> 标签，且顺序合理
        if pattern.search(completion):
            rewards.append(1.0)  # 格式正确
        else:
            rewards.append(-0.5)  # 格式错误，施加惩罚

    return rewards


def accuracy_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    奖励模型提取的最终答案是否与数据集中的 'solution' 匹配。
    - 奖励值: +3.0 (正确答案) / 0.0 (错误答案)
    """
    rewards = []

    # 确保数据集中的 'solution' 字段存在
    solutions = kwargs.get('solution')
    if not solutions or len(solutions) != len(completions):
        print("Warning: 'solution' field missing or mismatch in length for accuracy_reward_func.")
        # 如果无法获取正确答案，则不提供奖励信号
        return [0.0] * len(completions)

    for completion, solution in zip(completions, solutions):
        # 1. 提取模型答案
        model_answer = parse_answer_tag(completion)

        # 2. 规范化正确答案 (确保格式一致性，这是最难的部分)
        target_solution = str(solution).strip()

        # 3. 检查匹配
        # 注意: 这里的匹配必须非常严格 (例如，数字比较)
        if model_answer == target_solution:
            rewards.append(3.0)  # 高度奖励正确答案
        else:
            rewards.append(0.0)

    return rewards


def steps_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    奖励模型在 <think> 标签中生成清晰、结构化的推理步骤。
    - 奖励值: 基于检测到的推理步骤数量。
    """
    rewards = []

    step_pattern = re.compile(r'(\s*[*-]\s|\s*\d+\.\s|Step\s\d+)', re.IGNORECASE)

    for completion in completions:
        # 仅在 <think> 标签内查找步骤
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)

        if think_match:
            think_content = think_match.group(1)
            # 统计步骤模式出现的次数
            step_count = len(step_pattern.findall(think_content))
            # 给予奖励：每检测到一个步骤，奖励 0.2 分 (例如，限制最大奖励为 1.0)
            reward = min(step_count * 0.2, 1.0)
            rewards.append(reward)
        else:
            rewards.append(-0.1)  # 缺少 <think> 标签，轻微惩罚

    return rewards


########## custom
from models.aime_model import AIMEModel_FREE
from models.medqa_model import MedQAModel_MCQ
from models.sum_model import SumModel_FREE
from models.diagnosis_model import DiagModel_FREE, DiagModel_MCQ
from models.gsm8k_model import GSM8KModel_FREE
from models.math_model import MATHModel_FREE
from typing import List, Optional, Union

aime_model = None
gsm8k_model = None
math_model = None
medqa_model = None
diag_model = None



# ==============================================================================

def medqa_acc_reward_func(completions, answer, **kwargs) -> List[float]:
    """
    答案正确性奖励函数（完全复用原类的reward_correct方法）
    输入格式：completions=模型输出列表, ground_truth=标准答案列表
    输出格式：与completions长度一致的奖励值列表（1.0=正确, -1.0=错误/无答案）
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        # 抽取内容
        completion = completion[0]['content']
        # 构造原reward_correct需要的item（item的"A"字段对应标准答案）
        item = {"A": gt.strip()}
        # 直接复用原类的reward_correct方法
        score = medqa_model.reward_correct(item, completion)
        rewards.append(score)
    return rewards

def medqa_format_reward_func(completions, answer, **kwargs) -> List[float]:
    """
    格式合规性奖励函数（完全复用原类的reward_format方法）
    输入格式：completions=模型输出列表, ground_truth=标准答案列表（仅占位，原方法未用到）
    输出格式：与completions长度一致的奖励值列表（1.5=boxed最优, -1.0=无有效答案）
    """
    rewards = []
    for completion in completions:
        # 抽取内容
        completion = completion[0]['content']
        # 构造空item（原reward_format方法未使用item参数，仅为兼容参数格式）
        item = {}
        # 直接复用原类的reward_format方法
        score = medqa_model.reward_format(item, completion)
        rewards.append(score)
    return rewards


def combined_medqa_reward_func(completions: List[str], answer: List[Union[str, int]], **kwargs) -> List[float]:
    """
    组合奖励函数（正确率0.7 + 格式0.3），总奖励归一化到0-1
    """
    # 获取各维度奖励
    acc_rewards = medqa_acc_reward_func(completions, answer, **kwargs)
    fmt_rewards = medqa_format_reward_func(completions, answer, **kwargs)

    # 加权融合（权重可调整）
    combined = [
        (acc * 0.7) + (fmt * 0.3)
        for acc, fmt in zip(acc_rewards, fmt_rewards)
    ]

    return combined

aime_model = None
def get_deepseek_r1_reward_funcs(data_name, task_name):
    # 全局实例化AIMEModel（复用所有方法）
    """返回用于 GRPOTrainer 的奖励函数列表。"""
    if data_name in ['GPQA'] + ['MMLU', 'MMLU-Pro', 'MedBullets', 'MedExQA', 'MedMCQA','MedQA', 'PubMedQA', 'AfrimedQA', 'MedxpertQA-R', 'MedxpertQA-U']: # medagents bench;
        global medqa_model
        if task_name == 'MCQ':
            medqa_model = MedQAModel_MCQ()
            return combined_medqa_reward_func
        else:
            raise NotImplementedError(f"Task {task_name} not supported.")
    else:
        raise NotImplementedError(f"Dataset {data_name} not supported for reward functions.")
