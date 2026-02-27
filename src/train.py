# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import subprocess
import torch
from models import AIMEModel_FREE, MedQAModel_MCQ, SumModel_FREE, GSM8KModel_FREE, DiagModel_MCQ, MATHModel_FREE, \
    PureLLMBaseline, DiagModel_FREE
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from functools import partial  # 导入partial用于绑定参数
from utils import print_trainable_parameters, preprocess_sft_function, preprocess_grpo_function, \
    preprocess_ppo_function, set_logging_file
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
from trl import PPOConfig, PPOTrainer
from reward import get_deepseek_r1_reward_funcs
from datasets import Dataset


def create_pretrain_model(train_dataset, set_config, adapter_root, output_root, ttl_root, template_config,
                          retrain=False, benchmark_name='base'):
    """sft/rl lora pretrain"""
    # https://zhuanlan.zhihu.com/p/722262401
    if not retrain and os.path.exists(output_root):
        print("Loading existing pre-trained model from:", output_root)
        set_config['MODEL_PATH'] = output_root
        if not os.path.exists(set_config['MODEL_PATH'] + '/config.json'):
            print("No valid tensor! Please retrain the model.")
            raise FileNotFoundError
    else:
        print("Starting pre-training LLM with LoRA...")
        train_llm(train_dataset, set_config, adapter_root, output_root)
        set_config['MODEL_PATH'] = output_root
    return create_pure_model(set_config, ttl_root, template_config, benchmark_name)


def create_pure_model(set_config, ttl_root, template_config, benchmark_name='base'):
    if set_config['MODEL'] not in ['TTL', 'TTT', 'SLOT', 'ours']:
        our_model = PureLLMBaseline(set_config['TASK'])
        our_model.data_name = set_config['DATASET']
        our_model.model_name = set_config['LLM']
        our_model.root = ttl_root
        our_model.tem_config = template_config
        our_model.tokenizer = AutoTokenizer.from_pretrained(set_config['MODEL_PATH'])
        if not set_config['PARALLEL']:
            our_model.load_model(set_config['MODEL_PATH'],
                                 device=set_config['DEVICE'])  # core model load，这里已经叠加了装饰器
        else:
            our_model.model_path = set_config['MODEL_PATH']
        our_model.tokenizer = AutoTokenizer.from_pretrained(set_config['MODEL_PATH'])  # 分词器加载

        masked_token_ids = [
            our_model.tokenizer.encode(token, add_special_tokens=False)[0]
            for token in ["system", "user", "assistant", ":", "\n"]
        ] if set_config['MASK_SPECIAL'] else None

        generation_params = GenerationConfig(
            do_sample=set_config['DO_SAMPLE'],
            temperature=set_config['TEMP'] if set_config['DO_SAMPLE'] else None,
            top_p=0.95 if set_config['DO_SAMPLE'] else None,
            max_new_tokens=set_config['MAX_LEN'],
            masked_token_ids=masked_token_ids
        )
        log_file = set_logging_file(set_config['MODEL_PATH'], ttl_root, benchmark_name)

        return log_file, our_model, None, generation_params

    """直接加载一个纯LLM模型"""
    if set_config['DATASET'] in ['AIME2024', 'AIME2025']:
        if set_config['TASK'] == 'FREE':
            our_model = AIMEModel_FREE()
        else:
            raise NotImplementedError(f"{set_config['DATASET']} task {set_config['TASK']} not supported yet.")
    elif set_config['DATASET'] in ['GSM8K']:
        if set_config['TASK'] == 'FREE':
            our_model = GSM8KModel_FREE()
        else:
            raise NotImplementedError(f"{set_config['DATASET']} task {set_config['TASK']} not supported yet.")
    elif set_config['DATASET'] in ['MATH500', 'HMMT2025', 'AMC']:
        if set_config['TASK'] == 'FREE':
            our_model = MATHModel_FREE()
        else:
            raise NotImplementedError(f"{set_config['DATASET']} task {set_config['TASK']} not supported yet.")
    elif set_config['DATASET'] in ['GPQA'] + ['MMLU', 'MMLU-Pro', 'MedBullets', 'MedExQA', 'MedMCQA', 'MedQA',
                                              'PubMedQA', 'AfrimedQA', 'MedxpertQA-R', 'MedxpertQA-U']:
        if set_config['TASK'] == 'MCQ':
            our_model = MedQAModel_MCQ()
        else:
            raise NotImplementedError(f"{set_config['DATASET']} task {set_config['TASK']} not supported yet.")
    elif set_config['DATASET'] in ['PLOS', 'eLife', 'Cochrane', 'SumPubmed', 'MedQsum', 'ACI-Bench', 'MTS-Diag']:
        if set_config['TASK'] == 'FREE':
            our_model = SumModel_FREE()
        else:
            raise NotImplementedError(f"{set_config['DATASET']} task {set_config['TASK']} not supported yet.")
    elif set_config['DATASET'] in ['DiagnosisArena', 'ReDis', 'CupCase', 'MediQ', 'PubHealth'] + ['LOGIQA', 'QASC',
                                                                                                  'ReClor']:
        if set_config['TASK'] == 'MCQ':
            our_model = DiagModel_MCQ()
        elif set_config['TASK'] == 'FREE':
            our_model = DiagModel_FREE()
        else:
            raise NotImplementedError(f"{set_config['DATASET']} task {set_config['TASK']} not supported yet.")
    else:
        raise NotImplementedError(f"Dataset {set_config['DATASET']} not supported yet.")
    args = our_model.setup_args(set_config)

    # new adds
    args.root = ttl_root

    # Setup environment and logging
    our_model.setup_environment(args)
    log_file = our_model.setup_logging(args, benchmark_name=benchmark_name)

    our_model.root = ttl_root
    our_model.ttl_state = set_config['TTL_STATE']
    our_model.model_name = set_config['LLM']
    our_model.tem_config = template_config
    our_model.tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if not args.parallel:
        our_model.load_model(args.model_path,
                             device=args.device)
    else:
        our_model.model_path = args.model_path

    our_model.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    masked_token_ids = [
        our_model.tokenizer.encode(token, add_special_tokens=False)[0]
        for token in ["system", "user", "assistant", ":", "\n"]
    ] if args.mask_special_tokens else None
    generation_params = GenerationConfig(
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=0.95 if args.do_sample else None,
        topk=20 if args.do_sample else None,
        min_p=0,
        max_new_tokens=args.max_new_tokens,
        masked_token_ids=masked_token_ids
    )
    return log_file, our_model, args, generation_params


def train_llm(train_dataset, set_config, adapter_root, output_root):
    # https://zhuanlan.zhihu.com/p/722262401
    # llm load
    print("Loading base model and tokenizer from:", set_config['LLM'])
    train_opt = set_config['PRETRAIN_OP']
    tokenizer = AutoTokenizer.from_pretrained(set_config['LLM_PATH'])

    model = AutoModelForCausalLM.from_pretrained(set_config['LLM_PATH'],
                                                 attn_implementation="flash_attention_2",
                                                 trust_remote_code=True,
                                                 
                                                 dtype=torch.bfloat16,
                                                 )
    model.config.use_cache = False  # 训练时关闭 cache
    # 很多模型（如 Llama/Mistral）需要设置 pad_token
    tokenizer.pad_token = tokenizer.eos_token
   
    # load lora config
    lora_config = LoraConfig(
        r=set_config['PRETRAIN_LORA_RANK'],
        lora_alpha=set_config['PRETRAIN_LORA_ALPHA'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 根据模型架构调整
        lora_dropout=set_config['PRETRAIN_LORA_DROP'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    train_data = train_dataset  # .data

    import math
    sample_size = 300  # math.floor(len(train_dataset) * 0.2)
    print("Pretrain Sample Size", sample_size)
    train_data = train_dataset[:sample_size]

    train_dataset = Dataset.from_list(train_data)
    print("Train dataset size", len(train_dataset))

    # training args
    if train_opt == 'SFT':
        print("Using SFT training...")
        # 定义训练参数
        training_args = SFTConfig(
            output_dir=adapter_root,
            overwrite_output_dir=True,
            warmup_ratio=0.1,
            num_train_epochs=set_config['PRETRAIN_EPOCHS'],
            per_device_train_batch_size=set_config['PRETRAIN_BATCH_SIZE'],
            gradient_accumulation_steps=set_config['GRADIENT_BACK'],
            lr_scheduler_type="cosine",
            save_steps=500,
            save_total_limit=2,
            logging_dir=adapter_root,  # tensorboard日志输出
            logging_steps=100,
            learning_rate=set_config['PRETRAIN_LR'],
            weight_decay=set_config['PRETRAIN_WD'],
            bf16=True,
            max_length=set_config['MAX_SFT_LEN'],
        )

        # 创建数据整理器
        # 应用预处理
        print("正在预处理SFT数据...")
        train_dataset = train_dataset.map(
            preprocess_sft_function,
            # batched=True,
            remove_columns=train_dataset.column_names,
            desc="Non-Tokenizing dataset"
        )
        print(f"预处理后的SFT数据集大小: {len(train_dataset)}")
        print(f"预处理后的SFT数据集特征: {train_dataset.features}")
        print("text字段内容示例：", train_dataset[0])

        # 创建 SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
           
        )
    elif train_opt == 'GRPO':
        print("Using GRPO training...")
        reward_funcs = get_deepseek_r1_reward_funcs(set_config['DATASET'], set_config['TASK'])
        training_args = GRPOConfig(
            output_dir=adapter_root,
            overwrite_output_dir=True,
            warmup_ratio=0.1,
            num_train_epochs=set_config['PRETRAIN_EPOCHS'],
            per_device_train_batch_size=set_config['PRETRAIN_BATCH_SIZE'],
            gradient_accumulation_steps=set_config['GRADIENT_BACK'],
            lr_scheduler_type="cosine",
            save_steps=500,
            save_total_limit=2,
            logging_dir=adapter_root,
            logging_steps=100,
            learning_rate=set_config['PRETRAIN_LR'],
            weight_decay=set_config['PRETRAIN_WD'],
            bf16=True,  # bf 16范围更大，精度有损失
            num_generations=set_config['PRETRAIN_GEN_NUM'],  #
            max_completion_length=set_config['MAX_GEN_LEN'],  # 完成长度
            use_vllm=True,
            # vllm_mode="server",
            # vllm_server_port=8001, # default 8000, 注意查看slurm脚本
            vllm_mode="colocate",  # 不然使用"server"； 需要提前启动，trl vllm-serve --model <model_name>
            vllm_gpu_memory_utilization=0.4,
            vllm_max_model_length=10000,  # 在最新的vllm，trl中使用
            vllm_tensor_parallel_size=1,
            generation_kwargs={
                "max_tokens": 2048,  # 控制生成的最大长度
                "temperature": 0.7,  # 控制生成的随机性
                "top_p": 0.9,  # Nucleus sampling 参数
                # "use_tools": False,
                # "tool_choice": None,
            }
        )
        print("正在预处理GRPO数据...")
        train_dataset = train_dataset.map(
            preprocess_grpo_function,
            # batched=True,
            remove_columns=train_dataset.column_names,  # 移除原始列，只保留模型需要的
            desc="Non-Tokenizing dataset"
        )
        # 找到你加载train_dataset的代码位置，添加以下检查
        print("text字段内容示例：", train_dataset[0])

        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
            reward_funcs=reward_funcs,
        )

    elif train_opt == 'PPO':
        print("Using PPO training...")
        print(f"Please note change the OUTPUT_PATH {output_root} in sh.")
        bash_script_path = "./ppo_train.sh"
        try:
            result = subprocess.run(
                bash_script_path,
                shell=True,
                check=True,
                capture_output=True,  # 捕获标准输出和标准错误
                text=True  # 将输出转为字符串（而非字节）
            )

            # 打印脚本执行结果
            print("脚本执行成功！")
            print("标准输出：", result.stdout)
            print("返回码：", result.returncode)  # 成功执行返回 0

        except subprocess.CalledProcessError as e:
            print(f"脚本执行失败！返回码：{e.returncode}")
            print("错误输出：", e.stderr)
        return

    else:
        raise NotImplementedError(f"Pretrain option {train_opt} not supported yet.")
    # 开始训练
    trainer.train()

    # 保存Lora adapter
    trainer.model.save_pretrained(adapter_root + '/final_lora_adapters')
    tokenizer.save_pretrained(adapter_root + '/final_lora_adapters')

    del trainer
    torch.cuda.empty_cache()  # 防止显存泄漏

    # 合并参数
    base_model = AutoModelForCausalLM.from_pretrained(set_config['LLM_PATH'],
                                                      trust_remote_code=True,
                                                      return_dict=True,
                                                      dtype=torch.bfloat16,  # 或 torch.bfloat16
                                                      device_map="auto")

    # 加载并合并 LoRA 权重
    model_to_merge = PeftModel.from_pretrained(base_model, adapter_root + '/final_lora_adapters')
    merged_model = model_to_merge.merge_and_unload()

    # 保存合并后的模型和分词器
    merged_model.save_pretrained(output_root)
    tokenizer.save_pretrained(output_root)
    print("All pretrain processs done!")

    torch.cuda.empty_cache()  # 防止显存泄漏


def load_lora_model(lora_path, set_config):
    """加载lora模型"""
    base_model = AutoModelForCausalLM.from_pretrained(set_config['LLM_PATH'], trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model


def purellm_evaluation(log_file, our_model, set_config, generation_params, test_dataset, voting=False):
    # load data
    our_model.eval_dataset = test_dataset
    # Run evaluation
    if set_config['AVG'] > 1:
        print(f"Running evaluation {set_config['AVG']} times and taking average...")
        accuracies = []
        format_accuracies = []
        other_metrics = []

        for run in range(set_config['AVG']):
            print(f"Run {run + 1}/{set_config['AVG']}...")
            if set_config['PARALLEL']:
                accuracy, format_accuracy, average_metrics = our_model.evaluate_model_parallel(
                    generation_params=generation_params,
                    seed=set_config['SEED'],
                    log_file=log_file,
                    max_parallel_gpus=set_config['MAX_PARALLEL_GPUS'],
                    voting=voting,
                )
            else:
                accuracy, format_accuracy, average_metrics = our_model.evaluate_model(
                    generation_params=generation_params,
                    seed=set_config['SEED'],
                    log_file=log_file,
                    voting=voting,
                )

            accuracies.append(accuracy)
            format_accuracies.append(format_accuracy)
            other_metrics.append(average_metrics)
            print(f"Run {run + 1} - Accuracy: {accuracy:.4f}, Format Accuracy: {format_accuracy:.4f}")
            print(f"Run {run + 1} - Other Metrics: {average_metrics}")

        # Calculate averages
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_format_accuracy = sum(format_accuracies) / len(format_accuracies)
        avg_other_metrics = {k: round(sum(d[k] for d in other_metrics) / len(other_metrics), 4) for k in
                             (other_metrics[0].keys() if other_metrics else [])}
        # Log average results
        with open(log_file, "a") as f:
            f.write(f"\n=== AVERAGE RESULTS ({set_config['AVG']} runs) ===\n")
            f.write(f"Individual Accuracies: {[f'{acc:.4f}' for acc in accuracies]}\n")
            f.write(f"Individual Format Accuracies: {[f'{acc:.4f}' for acc in format_accuracies]}\n")
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Average Format Accuracy: {avg_format_accuracy:.4f}\n")
            f.write(f"Average Other Metrics: {avg_other_metrics}\n")
            f.write(
                f"Accuracy Std Dev: {(sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5:.4f}\n")
            f.write(
                f"Format Accuracy Std Dev: {(sum((x - avg_format_accuracy) ** 2 for x in format_accuracies) / len(format_accuracies)) ** 0.5:.4f}\n")
            f.write(f"Other Metrics Std Dev: " + "{" + ", ".join([
                                                                     f"{k}: {(sum((d[k] - avg_other_metrics[k]) ** 2 for d in other_metrics) / len(other_metrics)) ** 0.5:.4f}"
                                                                     for k in avg_other_metrics.keys()]) + "}\n")
        print(
            f"Average Accuracy: {avg_accuracy:.4f} (±{(sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5:.4f})")
        print(
            f"Average Format Accuracy: {avg_format_accuracy:.4f} (±{(sum((x - avg_format_accuracy) ** 2 for x in format_accuracies) / len(format_accuracies)) ** 0.5:.4f})")
        print(f"Average Other Metrics: {avg_other_metrics}")
        print(f"Other Metrics Std Dev: " + "{" + ", ".join(
            [f"{k}: {(sum((d[k] - avg_other_metrics[k]) ** 2 for d in other_metrics) / len(other_metrics)) ** 0.5:.4f}"
             for k in avg_other_metrics.keys()]) + "}")
    else:
        # Run evaluation once (original behavior)
        if set_config['PARALLEL']:
            print("Running parallel evaluation across multiple GPUs...")
            accuracy, format_accuracy, average_metrics = our_model.evaluate_model_parallel(
                generation_params=generation_params,
                seed=set_config['SEED'],
                log_file=log_file,
                max_parallel_gpus=set_config['MAX_PARALLEL_GPUS'],
                voting=voting,
            )

        else:
            print("Running sequential evaluation...")
            accuracy, format_accuracy, average_metrics = our_model.evaluate_model(
                generation_params=generation_params,
                seed=set_config['SEED'],
                log_file=log_file,
                voting=voting,
            )
    print(f"Acc: {accuracy}. Format Acc: {format_accuracy}", f"Other Metrics: {average_metrics}")
    print("Evaluation completed.")


def ttl_online_train(log_file, our_model, args, generation_params, test_dataset):
    # load data
    our_model.eval_dataset = test_dataset  # [{}]

    # Run evaluation multiple times and take average if specified
    if args.average > 1:
        print(f"Running evaluation {args.average} times and taking average...")
        accuracies = []
        format_accuracies = []
        other_metrics = []

        for run in range(args.average):
            print(f"Run {run + 1}/{args.average}...")

            if args.parallel:
                accuracy, format_accuracy, average_metrics = our_model.evaluate_model_parallel(
                    generation_params=generation_params,
                    seed=args.seed,
                    log_file=log_file,
                    version=args.version,
                    max_parallel_gpus=args.max_parallel_gpus,
                )
            else:
                accuracy, format_accuracy, average_metrics = our_model.evaluate_model(
                    generation_params=generation_params,
                    seed=args.seed,
                    log_file=log_file,
                    version=args.version,
                )

            accuracies.append(accuracy)
            format_accuracies.append(format_accuracy)
            other_metrics.append(average_metrics)
            print(f"Run {run + 1} - Accuracy: {accuracy:.4f}, Format Accuracy: {format_accuracy:.4f}")
            print(f"Run {run + 1} - Other Metrics: {average_metrics}")

        # Calculate averages
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_format_accuracy = sum(format_accuracies) / len(format_accuracies)
        avg_other_metrics = {k: round(sum(d[k] for d in other_metrics) / len(other_metrics), 4) for k in
                             (other_metrics[0].keys() if other_metrics else [])}

        # Log average results
        with open(log_file, "a") as f:
            f.write(f"\n=== AVERAGE RESULTS ({args.average} runs) ===\n")
            f.write(f"Individual Accuracies: {[f'{acc:.4f}' for acc in accuracies]}\n")
            f.write(f"Individual Format Accuracies: {[f'{acc:.4f}' for acc in format_accuracies]}\n")
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Average Format Accuracy: {avg_format_accuracy:.4f}\n")
            f.write(f"Average Other Metrics: {avg_other_metrics}\n")
            f.write(
                f"Accuracy Std Dev: {(sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5:.4f}\n")
            f.write(
                f"Format Accuracy Std Dev: {(sum((x - avg_format_accuracy) ** 2 for x in format_accuracies) / len(format_accuracies)) ** 0.5:.4f}\n")
            f.write(f"Other Metrics Std Dev: " + "{" + ", ".join([
                                                                     f"{k}: {(sum((d[k] - avg_other_metrics[k]) ** 2 for d in other_metrics) / len(other_metrics)) ** 0.5:.4f}"
                                                                     for k in avg_other_metrics.keys()]) + "}\n")
        print(
            f"Average Accuracy: {avg_accuracy:.4f} (±{(sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5:.4f})")
        print(
            f"Average Format Accuracy: {avg_format_accuracy:.4f} (±{(sum((x - avg_format_accuracy) ** 2 for x in format_accuracies) / len(format_accuracies)) ** 0.5:.4f})")
        print(f"Average Other Metrics: {avg_other_metrics}")
        print(f"Other Metrics Std Dev: " + "{" + ", ".join(
            [f"{k}: {(sum((d[k] - avg_other_metrics[k]) ** 2 for d in other_metrics) / len(other_metrics)) ** 0.5:.4f}"
             for k in avg_other_metrics.keys()]) + "}")
    else:
        # Run evaluation once (original behavior)
        if args.parallel:
            print("Running parallel evaluation across multiple GPUs...")
            accuracy, format_accuracy, average_metrics = our_model.evaluate_model_parallel(
                generation_params=generation_params,
                seed=args.seed,
                log_file=log_file,
                version=args.version,
                max_parallel_gpus=args.max_parallel_gpus,
            )

        else:
            print("Running sequential evaluation...")
            accuracy, format_accuracy, average_metrics = our_model.evaluate_model(
                generation_params=generation_params,
                seed=args.seed,
                log_file=log_file,
                version=args.version,
            )
    print(f"Acc: {accuracy}. Format Acc: {format_accuracy}", f"Other Metrics: {average_metrics}")
    print(f"Evaluation complete. Results logged to {log_file}")


def ttl_offline_train(log_file, our_model, args, generation_params, test_dataset, voting=False):
    # load data
    our_model.eval_dataset = test_dataset

    # Run evaluation multiple times and take average if specified
    if args.average > 1:
        print(f"Running evaluation {args.average} times and taking average...")
        accuracies = []
        format_accuracies = []
        other_metrics = []

        for run in range(args.average):
            print(f"Run {run + 1}/{args.average}...")
            if args.parallel:
                accuracy, format_accuracy, average_metrics = our_model.evaluate_model_parallel_off(
                    generation_params=generation_params,
                    seed=args.seed,
                    log_file=log_file,
                    version=args.version,
                    max_parallel_gpus=args.max_parallel_gpus,
                )
            else:
                accuracy, format_accuracy, average_metrics = our_model.evaluate_model_off(
                    generation_params=generation_params,
                    seed=args.seed,
                    log_file=log_file,
                    version=args.version,
                )

            accuracies.append(accuracy)
            format_accuracies.append(format_accuracy)
            other_metrics.append(average_metrics)
            print(f"Run {run + 1} - Accuracy: {accuracy:.4f}, Format Accuracy: {format_accuracy:.4f}")
            print(f"Run {run + 1} - Other Metrics: {average_metrics}")
        # Calculate averages
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_format_accuracy = sum(format_accuracies) / len(format_accuracies)
        avg_other_metrics = {k: round(sum(d[k] for d in other_metrics) / len(other_metrics), 4) for k in
                             (other_metrics[0].keys() if other_metrics else [])}
        # Log average results
        with open(log_file, "a") as f:
            f.write(f"\n=== AVERAGE RESULTS ({args.average} runs) ===\n")
            f.write(f"Individual Accuracies: {[f'{acc:.4f}' for acc in accuracies]}\n")
            f.write(f"Individual Format Accuracies: {[f'{acc:.4f}' for acc in format_accuracies]}\n")
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Average Format Accuracy: {avg_format_accuracy:.4f}\n")
            f.write(f"Average Other Metrics: {avg_other_metrics}\n")
            f.write(
                f"Accuracy Std Dev: {(sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5:.4f}\n")
            f.write(
                f"Format Accuracy Std Dev: {(sum((x - avg_format_accuracy) ** 2 for x in format_accuracies) / len(format_accuracies)) ** 0.5:.4f}\n")
            f.write(f"Other Metrics Std Dev: " + "{" + ", ".join([
                                                                     f"{k}: {(sum((d[k] - avg_other_metrics[k]) ** 2 for d in other_metrics) / len(other_metrics)) ** 0.5:.4f}"
                                                                     for k in avg_other_metrics.keys()]) + "}\n")
        print(
            f"Average Accuracy: {avg_accuracy:.4f} (±{(sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5:.4f})")
        print(
            f"Average Format Accuracy: {avg_format_accuracy:.4f} (±{(sum((x - avg_format_accuracy) ** 2 for x in format_accuracies) / len(format_accuracies)) ** 0.5:.4f})")
        print(f"Average Other Metrics: {avg_other_metrics}")
        print(f"Other Metrics Std Dev: " + "{" + ", ".join(
            [f"{k}: {(sum((d[k] - avg_other_metrics[k]) ** 2 for d in other_metrics) / len(other_metrics)) ** 0.5:.4f}"
             for k in avg_other_metrics.keys()]) + "}")
    else:
        # Run evaluation once (original behavior)
        if args.parallel:
            print("Running parallel evaluation across multiple GPUs...")
            accuracy, format_accuracy, average_metrics = our_model.evaluate_model_parallel_off(
                generation_params=generation_params,
                seed=args.seed,
                log_file=log_file,
                version=args.version,
                max_parallel_gpus=args.max_parallel_gpus,
            )
        else:
            print("Running sequential evaluation...")
            accuracy, format_accuracy, average_metrics = our_model.evaluate_model_off(
                generation_params=generation_params,
                seed=args.seed,
                log_file=log_file,
                version=args.version,
            )
    print(f"Acc: {accuracy}. Format Acc: {format_accuracy}", f"Other Metrics: {average_metrics}")
    print(f"Evaluation complete. Results logged to {log_file}")


