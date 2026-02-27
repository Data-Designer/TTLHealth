# !/usr/bin/env python
# -*-coding:utf-8 -*-

import re
import argparse
import os
import random
import json
import time
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from .dsi_decarator import enable_dsi
from typing import List, Dict, Any, Tuple
from langchain_core.prompts import PromptTemplate
from datasets import Dataset


class BaseModel:
    def __init__(self):
        # Initialize
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.eval_dataset = None
        self.root = None
        self.tem_config = None  # prompt template
        self.ttl_state = None

    @staticmethod
    def detect_available_gpus() -> List[int]:
        """Detect available GPUs and return their indices"""
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU")
            return []

        gpu_count = torch.cuda.device_count()
        available_gpus = []

        for i in range(gpu_count):
            try:
                # Test if GPU is accessible
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                available_gpus.append(i)
                print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Available")
            except Exception as e:
                print(f"GPU {i}: Not available - {str(e)}")

        print(f"Found {len(available_gpus)} available GPUs: {available_gpus}")
        return available_gpus

    @staticmethod
    def partition_data(data: List[Dict], num_partitions: int) -> List[List[Dict]]:
        """Partition data into roughly equal chunks for parallel processing"""
        if num_partitions <= 1:
            data_with_idx = []
            for i, item in enumerate(data):
                item_with_idx = item.copy()
                item_with_idx['global_idx'] = i
                data_with_idx.append(item_with_idx)
            return [data_with_idx]

        chunk_size = len(data) // num_partitions
        remainder = len(data) % num_partitions

        partitions = []
        start_idx = 0

        for i in range(num_partitions):
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size

            partition_data = []
            for j, item in enumerate(data[start_idx:end_idx]):
                item_with_idx = item.copy()
                item_with_idx['global_idx'] = start_idx + j  # {'Q':x, 'global_ids':x}
                partition_data.append(item_with_idx)

            partitions.append(partition_data)
            start_idx = end_idx

        # Print partition info
        for i, partition in enumerate(partitions):  # [{},{},{}]
            print(
                f"Partition {i}: {len(partition)} samples (indices {partition[0]['global_idx']}-{partition[-1]['global_idx']})")

        return partitions

    @staticmethod
    def setup_args(config):
        """Setup command line arguments"""
        # default data
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str,
                            default=config['MODEL_PATH'],
                            help="Path to the model")
        parser.add_argument("--device", type=str, default='auto',
                            help="Device to run the model on (e.g., cuda:0, cpu)")  # "cuda:" + config['GPU']
        parser.add_argument("--eval_samples", type=int, default=None,
                            help="Number of samples to evaluate, None for full evaluation")
        parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                            help="Dataset split to evaluate on")

        # generation
        parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation")
        parser.add_argument("--temperature", type=float, default=config['TEMP'], help="Generation temperature")
        parser.add_argument("--max_new_tokens", type=int, default=config['MAX_LEN'],
                            help="Maximum number of new tokens to generate")
        parser.add_argument("--seed", type=int, default=config['SEED'],
                            help="Random seed for consistent evaluation samples")
        parser.add_argument("--use_entropy_control", action="store_true",
                            help="Enable entropy-based early stopping and continuation")
        parser.add_argument("--entropy_threshold", type=float, default=config['ENTROPY_THRE'],
                            help="Entropy threshold for early stopping")
        parser.add_argument("--long_entropy_threshold", type=float, default=config['LONG_ENTROPY_THRE'],
                            help="Entropy threshold for early stopping")

        parser.add_argument("--max_retries", type=int, default=config['MAX_RETRY'],
                            help="Maximum number of retries for entropy-controlled generation")
        # optimization
        parser.add_argument("--times", type=int, default=config['ITER_NUM'], help="Number of optimization iterations")
        parser.add_argument("--lr", type=float, default=config['LR'], help="Learning rate for optimization")
        parser.add_argument("--record_entropy", action="store_true", help="Whether to record entropy analysis")
        parser.add_argument("--entropy_output_file", type=str, default="my_analysis.jsonl",
                            help="Output file for entropy analysis")
        parser.add_argument("--query_weight", type=float, default=config['QUERY_WEIGHT'],
                            help="Weight for query ce loss")
        parser.add_argument("--structure_weight", type=float, default=config['STRU_WEIGHT'],
                            help="Weight for structure loss")
        parser.add_argument("--entropy_weight", type=float, default=config['ENTROPY_WEIGHT'],
                            help="Weight for entropy loss")
        parser.add_argument("--adaptive_entropy", action="store_true", help="Enable adaptive entropy threshold")
        parser.add_argument("--adaptive_entropy_N", type=int, default=config['ENTROPY_TOKEN_MIN'],
                            help="Number of samples for adaptive entropy threshold")  # tokens
        parser.add_argument("--adaptive_entropy_K", type=float, default=config['ENTROPY_WIN'],
                            help="K for adaptive entropy threshold")  # windows
        parser.add_argument("--long_adaptive_entropy_K", type=float, default=config['LONG_ENTROPY_WIN'],
                            help="K for adaptive entropy threshold")  # windows
        parser.add_argument("--long_adaptive_entropy_N", type=int, default=config['LONG_ENTROPY_TOKEN_MIN'],
                            help="Number of samples for adaptive entropy threshold")  # tokens

        parser.add_argument("--mask_special_tokens", action="store_true",
                            help="Mask special tokens in the input")
        parser.add_argument("--set_minimal_threshold", action="store_true",
                            help="Set minimal threshold for entropy control")
        parser.add_argument("--minimal_std", type=float, default=0.5, help="std for minimal threshold")
        parser.add_argument("--minimal_threshold", type=float, default=1.8, help="Threshold for minimal threshold")

        # Parallel evaluation arguments
        parser.add_argument("--parallel", action="store_true", help="Enable parallel evaluation across multiple GPUs")
        parser.add_argument("--max_parallel_gpus", type=int, default=None,
                            help="Maximum number of GPUs to use for parallel evaluation")

        # Average evaluation arguments
        parser.add_argument("--average", type=int, default=config['AVG'],
                            help="Number of times to run evaluation and take average")

        parser.add_argument("--version", type=str, help="Version of Same dataset")
        return parser.parse_args()

    @staticmethod
    def setup_environment(args):
        """Setup environment variables, 防止重新设置"""
        os.environ["times"] = str(args.times)
        os.environ["lr"] = str(args.lr)
        os.environ["record_entropy"] = str(args.record_entropy).lower()
        os.environ["entropy_output_file"] = args.root + '/' + args.entropy_output_file
        os.environ["tokenizer_path"] = args.model_path

        os.environ["entropy_threshold"] = str(args.entropy_threshold)
        os.environ["long_entropy_threshold"] = str(args.long_entropy_threshold)

        os.environ["entropy_weight"] = str(args.entropy_weight)
        os.environ["structure_weight"] = str(args.structure_weight)
        os.environ["query_weight"] = str(args.query_weight)

        os.environ["adaptive_entropy"] = "True" if args.adaptive_entropy else "False"
        os.environ["adaptive_entropy_N"] = str(args.adaptive_entropy_N)
        os.environ["adaptive_entropy_K"] = str(args.adaptive_entropy_K)
        os.environ["long_adaptive_entropy_N"] = str(args.long_adaptive_entropy_N)
        os.environ["long_adaptive_entropy_K"] = str(args.long_adaptive_entropy_K)
        os.environ["temperature"] = str(args.temperature) if args.do_sample else "1.0"

        if args.use_entropy_control:
            os.environ["use_entropy_control"] = "True"
            os.environ["entropy_threshold"] = str(args.entropy_threshold)
            os.environ["long_entropy_threshold"] = str(args.long_entropy_threshold)

            os.environ["max_retries"] = str(args.max_retries)

            os.environ["minimal_std"] = str(args.minimal_std)
            os.environ["minimal_threshold"] = str(args.minimal_threshold)

            print(f"Entropy control enabled with threshold: {args.entropy_threshold}, max retries: {args.max_retries}")
        else:
            os.environ["use_entropy_control"] = "False"

    @staticmethod
    def setup_logging(args, benchmark_name: str = "base"):
        """Setup logging directory and file，方便查看不同的超参数结果"""
        model_name = args.model_path.split("/")[-1]
        log_dir = args.root + '/' + f"logs/{benchmark_name}/{model_name}"
        os.makedirs(log_dir, exist_ok=True)
        max_retries = args.max_retries
        entropy_suffix = f"_entropy_{args.entropy_threshold}_weight_{args.entropy_weight}" if args.use_entropy_control else ""
        adaptive_entropy_suffix = f"_N_{args.adaptive_entropy_N}_K_{args.adaptive_entropy_K}" if args.adaptive_entropy else ""
        do_sample_suffix = f"_do_sample_temperature_{args.temperature}" if args.do_sample else ""

        mask_special_suffix = "" if args.mask_special_tokens else "_nomask"
        parallel_suffix = "_parallel" if getattr(args, 'parallel', False) else ""
        average_suffix = f"_avg_{args.average}" if args.average > 1 else ""

        log_file = os.path.join(log_dir,
                                f"log_{model_name}_times_{args.times}_lr_{args.lr}{entropy_suffix}{adaptive_entropy_suffix}_reatries_{max_retries}{do_sample_suffix}{mask_special_suffix}{parallel_suffix}{average_suffix}.txt")

        with open(log_file, "w") as f:
            f.write(f"Model Path: {args.model_path}\n")
            f.write(f"Times: {args.times}\n")
            f.write(f"LR: {args.lr}\n")
            f.write(f"Record Entropy: {args.record_entropy}\n")
            f.write(f"Entropy Output File: {args.entropy_output_file}\n")
            f.write(f"Entropy Weight: {args.entropy_weight}\n")
            f.write(f"Structure Weight: {args.structure_weight}\n")
            f.write(f"Query Weight: {args.query_weight}\n")
            f.write(f"Eval Samples: {'All' if args.eval_samples is None else args.eval_samples}\n")
            f.write(f"Dataset Split: {args.split}\n")
            f.write(f"Do Sample: {args.do_sample}\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Use Entropy Control: {args.use_entropy_control}\n")
            f.write(f"Entropy Threshold: {args.entropy_threshold}\n")
            f.write(f"Max Retries: {args.max_retries}\n")
            f.write(f"Parallel Evaluation: {getattr(args, 'parallel', False)}\n")
            if getattr(args, 'parallel', False):
                f.write(f"Max Parallel GPUs: {getattr(args, 'max_parallel_gpus', 'All available')}\n")
            f.write(f"Average Runs: {args.average}\n")
            f.write("\n")

        return log_file

    def is_phi_model(self):
        """Check if the current model is a Phi model"""
        if self.model is None:
            return False

        # Check model type from config
        model_type = getattr(self.model.config, 'model_type', '').lower()
        if model_type in ['phi', 'phi3']:
            return True

        # Check model class name
        class_name = self.model.__class__.__name__.lower()
        if 'phi' in class_name:
            return True

        return False

    def load_model(self, model_path, device="auto"):
        """Load model and tokenizer with automatic model type detection"""
        print(f"Loading model from: {model_path}")
        self.model_path = model_path  # Store for parallel processes
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = AutoConfig.from_pretrained(model_path)
        model_type = config.model_type.lower()

        print(f"Detected model type: {model_type}")

        DSIModelClass = enable_dsi(AutoModelForCausalLM)
        print("Change to the DSI!")

        print(f"Loading model with universal DSI implementation (model_type: {model_type})...")
        self.model = DSIModelClass.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=device,
            trust_remote_code=True
        )

        if self.is_phi_model():
            print("Phi model detected: Will use combined system+user prompt format")

    def build_prompt_text(self, sys_prompt, prompt):
        """Build prompt text"""
        prompt_text = self.tokenizer.apply_chat_template([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ], tokenize=False, add_generation_prompt=True)
        return prompt_text

    def create_ttt_dataset(
            self,
            correct_examples: list,
            num_training_steps: int,
            dataset_filename: str,
            shuffle_examples: bool,
            reverse: bool = False,
            ignore_pseudo: bool = True
    ):
        """
        Create a multi-sample JSON dataset for Torchtune finetuning, with no chain-of-thought.

        - If shuffle_examples=True, each item will be a random permutation of correct_examples.
        - If shuffle_examples=False, each item uses the same order of correct_examples.

        We prepend `prefix` to each sample. Then, for each (Q, A) pair, we append:
           Q: ...
           A: ...
        """
        random_gate = random.random() < 0.9
        if ignore_pseudo and random_gate:
            correct_examples = correct_examples[:-1]
            syn_examples = [correct_examples[-1]]
        data_samples = []

        for _ in range(num_training_steps):
            if shuffle_examples:
                perm = correct_examples[:]
                random.shuffle(perm)
                ex_list, target = perm[:-1], perm[-1]
                if ignore_pseudo and random_gate:

                    ex_list.extend(syn_examples)
                    random.shuffle(ex_list)
            else:
                ex_list, target = correct_examples[:-1], correct_examples[-1]

            if reverse:
                text = self._formulate_dem([(a, q) for (q, a) in ex_list], self.tem_config['inv_dem'])
            else:
                text = self._formulate_dem(ex_list, self.tem_config['dem'])
            data_samples.append({"demonstrations": text, "target_Q": target[0], "target_A": target[1]})

        with open(self.root + '/' + dataset_filename, 'w') as f:
            json.dump(data_samples, f)
        return data_samples

    def extract_contexts(self, text: str, strip_whitespace: bool = False) -> str:
        """demonstration和question的分隔符是your turn"""
        pattern = r"^(.*?)\s*Your"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            result = match.group(1)
            if strip_whitespace:
                result = result.strip()
            return result
        return ""

    def _formulate_dem(self, retrival_icl, demonstration_tem):
        demonstrations = ''
        for id, (q, a) in enumerate(retrival_icl):
            demonstrations += PromptTemplate.from_template(demonstration_tem).format(
                id=id + 1,
                question=q,
                answer=a,
            )
        return demonstrations

    def build_structure_text(self, sys_prompt, sys_prompt_inverse, demonstrations, syn_pair, inverse=False):
        """Build prompt text"""
        demonstrations = demonstrations + [syn_pair]
        structure_list = self.create_ttt_dataset(demonstrations,
                                                 num_training_steps=len(demonstrations),
                                                 dataset_filename="temp_ttt_dataset.json",
                                                 shuffle_examples=True)

        # 进行编码
        structure_id_list = []
        for prompt_dic in structure_list:
            user_prompt = PromptTemplate.from_template(self.tem_config['tes'])
            prompt = user_prompt.format(question=prompt_dic['target_Q'],
                                        demonstration=prompt_dic['demonstrations'])

            prompt_text = self.tokenizer.apply_chat_template([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)  # 加上了assisant
            structure_id_list.append({'text': prompt_text, 'label': prompt_dic['target_A']})

        if inverse:
            demonstrations = [(a, q) for (q, a) in demonstrations]  # 交换
            inverse_structure_list = self.create_ttt_dataset(demonstrations,
                                                             num_training_steps=len(demonstrations),  # N次
                                                             dataset_filename="temp_ttt_dataset_inverse.json",
                                                             shuffle_examples=True)
            inverse_structure_id_list = []
            for prompt in inverse_structure_list:
                # Standard format for other models
                user_prompt = PromptTemplate.from_template(self.tem_config['inv_tes'])
                prompt = user_prompt.format(solution=prompt['target_A'],
                                            demonstration=prompt['demonstrations'])

                prompt_text = self.tokenizer.apply_chat_template([
                    {"role": "system", "content": sys_prompt_inverse},
                    {"role": "user", "content": prompt}
                ], tokenize=False, add_generation_prompt=True)
                inverse_structure_id_list.append({'text': prompt_text, 'label': prompt['target_Q']})
        else:
            inverse_structure_id_list = []
        # 合并
        all_list = structure_id_list + inverse_structure_id_list
        random.shuffle(all_list)

        return all_list

    def format_and_tokenize_dataset(self, stru_list: list, target_device="cpu"):
        TEMPLATE = "{question}" + '\n' + self.tem_config[
            'format']

        def create_instruction_text(example):
            return {
                "text": TEMPLATE.format(
                    question=example["text"],
                    answer=example["label"],
                )
            }

        def tokenize_and_mask_labels(example):
            full_text = example["text"]
            response_start_marker = self.tem_config['format'][:7]

            response_start_index = full_text.rfind(response_start_marker)

            prompt_text = full_text[:response_start_index]
            tokenized_full = self.tokenizer(
                full_text,
                return_tensors="pt"
            )
            tokenized_prompt = self.tokenizer(
                prompt_text,
                return_tensors="pt"
            )

            input_ids = tokenized_full["input_ids"]
            labels = input_ids.clone()

            prompt_len = tokenized_prompt["input_ids"].shape[1]

            labels[:, :prompt_len] = -100

            return {
                "input_ids": tokenized_full["input_ids"].to(target_device),
                "labels": labels.to(target_device),
                "attention_mask": tokenized_full["attention_mask"].to(target_device)
            }

        raw_dataset = Dataset.from_list(stru_list)

        # 1. 格式化
        formatted_dataset = raw_dataset.map(create_instruction_text, remove_columns=["text", "label"])
        formatted_data_list = formatted_dataset.to_list()
        tokenized_lis = list(map(tokenize_and_mask_labels, formatted_data_list))
  
        # )
        return tokenized_lis

    def generate_with_entropy_control(self, inputs, generation_params, max_retries=5,
                                      context_len=0):
        """Generate text with entropy control, 这里不是adaptive的"""
        os.environ["entropy_control"] = "True"
        os.environ["log_entropy_control"] = "True"

        full_completion = ""
        current_inputs = inputs.copy()
        retry_count = 0

        while retry_count < max_retries:
            self.model.reset_entropy_detection()
            self.model.prompt_only = True

            con_query_mask = torch.zeros_like(current_inputs['input_ids']).to(self.model.device)
            con_query_mask[:, context_len:] = 1
            self.model.con_query_mask = con_query_mask

            outputs = self.model.generate(
                **current_inputs,
                generation_config=generation_params,
                # **generation_params,
            )

            new_tokens = outputs[0][current_inputs['input_ids'].shape[1]:]
            completion_part = self.tokenizer.decode(new_tokens,
                                                    skip_special_tokens=True)
            del outputs
            torch.cuda.empty_cache()

            if self.model.high_entropy_detected:  # 有个entropy的标准
                print(f"High entropy detected at retry {retry_count}, position {self.model.high_entropy_position}")
                print(f"Partial completion: {completion_part}")

                full_completion += completion_part

                old_inputs = current_inputs
                new_text = self.tokenizer.decode(current_inputs['input_ids'][0],
                                                 skip_special_tokens=True) + completion_part
                current_inputs = self.tokenizer(new_text, return_tensors="pt", add_special_tokens=False).to(
                    self.model.device)
                del old_inputs

                retry_count += 1
                print(f"Continuing generation with {current_inputs['input_ids'].shape[1]} tokens")
            else:
                full_completion += completion_part
                print(f"Generation completed normally after {retry_count} retries")
                break

        if retry_count >= max_retries:  #
            print(f"Max retries ({max_retries}) reached due to high entropy, continuing with normal generation")

            os.environ["entropy_control"] = "False"
            self.model.prompt_only = False

            self.model.reset_entropy_detection()

            print(f"Continuing normal generation from {current_inputs['input_ids'].shape[1]} tokens")
            final_outputs = self.model.generate(
                **current_inputs,
                generation_config=generation_params,
            )

            final_new_tokens = final_outputs[0][current_inputs['input_ids'].shape[1]:]
            final_completion_part = self.tokenizer.decode(final_new_tokens, skip_special_tokens=True)

            full_completion += final_completion_part
            print(f"Normal generation completed, added {len(final_new_tokens)} tokens")
        else:
            print(f"Generation completed normally after {retry_count} retries")

        os.environ["entropy_control"] = "False"

        return full_completion, retry_count

    def generate_with_entropy_control_off(self, inputs, generation_params, max_retries=1,
                                          context_len=0):
        os.environ["entropy_control"] = "True"
        os.environ["log_entropy_control"] = "True"

        full_completion = ""
        current_inputs = inputs.copy()
        retry_count = max_retries + 1

        while retry_count < max_retries:
            self.model.reset_entropy_detection()
            self.model.prompt_only = False

            con_query_mask = torch.zeros_like(current_inputs['input_ids']).to(self.model.device)
            con_query_mask[:, context_len:] = 1
            self.model.con_query_mask = con_query_mask

            outputs = self.model.generate(
                **current_inputs,
                generation_config=generation_params,
            )

            new_tokens = outputs[0][current_inputs['input_ids'].shape[1]:]
            completion_part = self.tokenizer.decode(new_tokens,
                                                    skip_special_tokens=True)  #

            del outputs
            torch.cuda.empty_cache()

            if self.model.high_entropy_detected:
                print(f"High entropy detected at retry {retry_count}, position {self.model.high_entropy_position}")
                print(f"Partial completion: {completion_part}")

                full_completion += completion_part

                old_inputs = current_inputs
                new_text = self.tokenizer.decode(current_inputs['input_ids'][0],
                                                 skip_special_tokens=True) + completion_part
                current_inputs = self.tokenizer(new_text, return_tensors="pt", add_special_tokens=False).to(
                    self.model.device)
                del old_inputs

                retry_count += 1
                print(f"Continuing generation with {current_inputs['input_ids'].shape[1]} tokens")
            else:
                full_completion += completion_part
                print(f"Generation completed normally after {retry_count} retries")
                break

        if retry_count >= max_retries:  # 如果搞了5次还不行，就把之前的一并输入，最后一次尝试了。
            print(f"Max retries ({max_retries}) reached due to high entropy, continuing with normal generation")

            os.environ["entropy_control"] = "False"
            self.model.prompt_only = False

            self.model.reset_entropy_detection()

            print(f"Continuing normal generation from {current_inputs['input_ids'].shape[1]} tokens")
            final_outputs = self.model.generate(
                **current_inputs,
                generation_config=generation_params,
            )

            final_new_tokens = final_outputs[0][current_inputs['input_ids'].shape[1]:]
            final_completion_part = self.tokenizer.decode(final_new_tokens, skip_special_tokens=True)

            full_completion += final_completion_part  # 完全的new text
            print(f"Normal generation completed, added {len(final_new_tokens)} tokens")
        else:
            print(f"Generation completed normally after {retry_count} retries")

        os.environ["entropy_control"] = "False"

        return full_completion, retry_count

    def evaluate_model(self, generation_params=None, seed=42,
                       log_file="evaluation_log.txt", version=None):
        """Evaluate model on dataset, core changes"""
        print("Starting model evaluation...")
        self.model.eval()
        random.seed(seed)

        # eval_QAs = self.load_dataset(split, eval_samples, version=version)
        eval_QAs = self.eval_dataset
        print(f"Evaluating {len(eval_QAs)} samples")

        with open(log_file, "a") as f:
            f.write(f"Number of evaluation samples: {len(eval_QAs)}\n\n")
            f.write(f"Start time: {time.time()}\n")

        correct = 0
        format_correct = 0
        other_metrics = []

        total = len(eval_QAs)
        total_retries = 0

        for i, qa in enumerate(eval_QAs):
            print(f"======================Sample {i}=====================")
            self.model.reset_entropy_detection()
            self.model.reset_model_parameters()
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{total} samples")

            sys_prompt, demonstrations, prompt, syn_pair = qa['system'], qa['retrieved_texts'], qa['prompt'], qa[
                'syn_pair']
            syn_pair = list(syn_pair)
            syn_pair[1] = syn_pair[1].split('</think>', 1)[1].strip() if '</think>' in syn_pair[1] else syn_pair[
                1]
            # semantics & second structure -> decorator
            prompt_text = self.build_prompt_text(sys_prompt, prompt)  # for semantics

            # structure list
            sys_prompt_inverse = self.tem_config['inv_sys']
            structure_text_lis = self.build_structure_text(sys_prompt, sys_prompt_inverse, demonstrations, syn_pair,
                                                           inverse=False)  # for

            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True).to(
                self.model.device)

            context = self.extract_contexts(prompt_text, strip_whitespace=False)

            context_len = self.tokenizer(context, return_tensors="pt", add_special_tokens=False, truncation=True).to(
                self.model.device)  # 如果非ICL的话，这里按理说是0
            context_len = context_len['input_ids'].shape[1]

            con_query_mask = torch.zeros_like(inputs['input_ids']).to(self.model.device)  # 默认全0，不mask任何token
            con_query_mask[:, context_len:] = 1
           

            struc_dataset = self.format_and_tokenize_dataset(structure_text_lis,
                                                             target_device=self.model.device)
            self.model.struc_dataset, self.model.con_query_mask = struc_dataset, con_query_mask
            self.model.ttl_state = self.ttl_state

     

            use_entropy_control = os.environ.get("use_entropy_control", "False") == "True"
            if use_entropy_control:
                print(f"\n--- Sample {i + 1} use_entropy_control start---")
                max_retries = int(os.environ.get("max_retries", "5"))
                completion, retry_count = self.generate_with_entropy_control(inputs, generation_params, max_retries,
                                                                             context_len)
                print(f"--- Sample {i + 1} use_entropy_control end---")
            else:
                print("--- No Entropy Control, Only with Optimization ---")
                self.model.prompt_only = True
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_params,
                )
                completion = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                                   skip_special_tokens=True)
                retry_count = 0

            # 使用后清理
            self.model.struc_dataset = None
            self.model.con_query_mask = None

            format_score = self.reward_format(qa, completion)  # 继承类实现
            correct_score = self.reward_correct(qa, completion)
            other_metrics_scores = self.reward_other_metrics(qa, completion)

            is_format_correct = format_score > 0
            is_answer_correct = correct_score > 0
            other_metrics.append(other_metrics_scores)

            if is_format_correct:
                format_correct += 1
            if is_answer_correct:
                correct += 1

            total_retries += retry_count

            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"Sample {i + 1}:\n")
                f.write(f'In context Question: {prompt_text}\n')
                f.write(f"Question: {qa['Q']}\n")
                f.write(f"Model Response: {completion}\n")
                f.write(f"Correct Answer: {qa['A']}\n")
                f.write(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}\n")
                f.write(f"Other Metric Score: {other_metrics_scores}\n")
                f.write(f"Retry Count: {retry_count}\n\n")

            print(f"\n--- Sample {i + 1} ---")
            print("Question:", qa['Q'])
            print("In context Question:", prompt_text)
            print("Model Response:", completion)
            print("Correct Answer:", qa['A'])
            print(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}")
            print(f"Other Metric Score: {other_metrics_scores}")
            print(f"Retry Count: {retry_count}")

        accuracy = correct / total if total > 0 else 0
        format_accuracy = format_correct / total if total > 0 else 0
        avg_retries = total_retries / total if total > 0 else 0
        average_metrics = {k: round(sum(d[k] for d in other_metrics) / len(other_metrics), 4) for k in
                           (other_metrics[0].keys() if other_metrics else [])}

        print(f"\nEvaluation Results (Samples: {total}):")
        print(f"Answer Accuracy: {accuracy:.4f}")
        print(f"Format Accuracy: {format_accuracy:.4f}")
        print(f"Other Metrics: {average_metrics}")
        print(f"Total Retries: {total_retries}")
        print(f"Average Retries per Sample: {avg_retries:.2f}")

        with open(log_file, "a") as f:
            f.write(f"End time: {time.time()}\n")
            f.write(f"Evaluation Results (Samples: {total}):\n")
            f.write(f"Answer Accuracy: {accuracy:.4f}\n")
            f.write(f"Format Accuracy: {format_accuracy:.4f}\n")
            f.write(f"Other Metrics: {average_metrics}\n")
            f.write(f"Total Retries: {total_retries}\n")
            f.write(f"Average Retries per Sample: {avg_retries:.2f}\n")

        return accuracy, format_accuracy, average_metrics

    def evaluate_model_off(self, generation_params=None, seed=42,
                           log_file="evaluation_log.txt", version=None):
        """Evaluate model on dataset, core changes"""
        print("Starting model evaluation...")
        self.model.eval()
        random.seed(seed)

        eval_QAs = self.eval_dataset
        print(f"Evaluating {len(eval_QAs)} samples")

        with open(log_file, "a") as f:
            f.write(f"Number of evaluation samples: {len(eval_QAs)}\n\n")
            f.write(f"Start time: {time.time()}\n")

        correct = 0
        format_correct = 0
        other_metrics = []

        total = len(eval_QAs)
        total_retries = 0
        self.model.reset_offline_parameters()
        for i, qa in enumerate(eval_QAs):
            self.model.reset_entropy_detection()
            self.model.reset_model_parameters()

            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{total} samples")

            sys_prompt, demonstrations, prompt, syn_pair = qa['system'], qa['retrieved_texts'], qa['prompt'], qa[
                'syn_pair']
            # semantics & second structure -> decorator
            prompt_text = self.build_prompt_text(sys_prompt, prompt)  # for semantics

            # structure list
            sys_prompt_inverse = self.tem_config['inv_sys']
            structure_text_lis = self.build_structure_text(sys_prompt, sys_prompt_inverse, demonstrations, syn_pair,
                                                           inverse=False)  # for structure

            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True).to(
                self.model.device)
     
            context = self.extract_contexts(prompt_text, strip_whitespace=False)
            context_len = self.tokenizer(context, return_tensors="pt", add_special_tokens=False, truncation=True).to(
                self.model.device)
            context_len = context_len['input_ids'].shape[1]

            con_query_mask = torch.zeros_like(inputs['input_ids']).to(self.model.device)  # 默认全0，不mask任何token
            con_query_mask[:, context_len:] = 1

            struc_dataset = self.format_and_tokenize_dataset(structure_text_lis,
                                                             target_device=self.model.device)
            self.model.struc_dataset, self.model.con_query_mask = struc_dataset, con_query_mask
            self.model.ttl_state = self.ttl_state

            use_entropy_control = os.environ.get("use_entropy_control", "False") == "True"
            if use_entropy_control:
                print(f"\n--- Sample {i + 1} use_entropy_control start---")
                max_retries = int(os.environ.get("max_retries", "5"))
                self.generate_with_entropy_control(inputs, generation_params, max_retries, context_len)
                print(f"--- Sample {i + 1} use_entropy_control end---")
            else:
                print("--- No Entropy Control, Only with Optimization ---")
                self.model.prompt_only = True
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_params,
                )

            self.model.struc_dataset = None
            self.model.con_query_mask = None

        for i, qa in enumerate(eval_QAs):
            self.model.reset_entropy_detection()
            self.model.reset_model_parameters()

            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{total} samples")

            sys_prompt, demonstrations, prompt, syn_pair = qa['system'], qa['retrieved_texts'], qa['prompt'], qa[
                'syn_pair']

            prompt_text = self.build_prompt_text(sys_prompt, prompt)  # for semantics

            # structure list
            sys_prompt_inverse = self.tem_config['inv_sys']
            structure_text_lis = self.build_structure_text(sys_prompt, sys_prompt_inverse, demonstrations, syn_pair,
                                                           inverse=False)  # for structure

            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True).to(
                self.model.device)

            context = self.extract_contexts(prompt_text, strip_whitespace=False)
            context_len = self.tokenizer(context, return_tensors="pt", add_special_tokens=False, truncation=True).to(
                self.model.device)
            context_len = context_len['input_ids'].shape[1]

            con_query_mask = torch.zeros_like(inputs['input_ids']).to(self.model.device)
            con_query_mask[:, context_len:] = 1

            struc_dataset = self.format_and_tokenize_dataset(structure_text_lis,
                                                             target_device=self.model.device)
            self.model.struc_dataset, self.model.con_query_mask = struc_dataset, con_query_mask
            self.model.ttl_state = self.ttl_state

            use_entropy_control = os.environ.get("use_entropy_control", "False") == "True"
            if use_entropy_control:
                print(f"\n--- Sample {i + 1} use_entropy_control start---")
                max_retries = int(os.environ.get("max_retries", "5"))
                completion, retry_count = self.generate_with_entropy_control_off(inputs, generation_params, max_retries,
                                                                                 context_len)
                print(f"--- Sample {i + 1} use_entropy_control end---")
            else:  # 不控制，这是baseline
                print("--- No Entropy Control, Only with Optimization ---")
                self.model.prompt_only = False  # 不优化，直接输出
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_params,
                )
                completion = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                                   skip_special_tokens=True)
                retry_count = 0

            # 使用后清理
            self.model.struc_dataset = None
            self.model.con_query_mask = None

            format_score = self.reward_format(qa, completion)  # 继承类实现
            correct_score = self.reward_correct(qa, completion)
            other_metrics_scores = self.reward_other_metrics(qa, completion)

            is_format_correct = format_score > 0
            is_answer_correct = correct_score > 0
            other_metrics.append(other_metrics_scores)

            if is_format_correct:
                format_correct += 1
            if is_answer_correct:
                correct += 1

            total_retries += retry_count

            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"Sample {i + 1}:\n")
                f.write(f'In context Question: {prompt_text}\n')
                f.write(f"Question: {qa['Q']}\n")
                f.write(f"Model Response: {completion}\n")
                f.write(f"Correct Answer: {qa['A']}\n")
                f.write(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}\n")
                f.write(f"Other Metric Score: {other_metrics_scores}\n")
                f.write(f"Retry Count: {retry_count}\n\n")

            print(f"\n--- Sample {i + 1} ---")
            print("Question:", qa['Q'])
            print("In context Question:", prompt_text)
            print("Model Response:", completion)
            print("Correct Answer:", qa['A'])
            print(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}")
            print(f"Other Metric Score: {other_metrics_scores}")
            print(f"Retry Count: {retry_count}")

        accuracy = correct / total if total > 0 else 0
        format_accuracy = format_correct / total if total > 0 else 0
        avg_retries = total_retries / total if total > 0 else 0
        average_metrics = {k: round(sum(d[k] for d in other_metrics) / len(other_metrics), 4) for k in
                           (other_metrics[0].keys() if other_metrics else [])}

        print(f"\nEvaluation Results (Samples: {total}):")
        print(f"Answer Accuracy: {accuracy:.4f}")
        print(f"Format Accuracy: {format_accuracy:.4f}")
        print(f"Other Metrics: {average_metrics}")
        print(f"Total Retries: {total_retries}")
        print(f"Average Retries per Sample: {avg_retries:.2f}")

        with open(log_file, "a") as f:
            f.write(f"End time: {time.time()}\n")
            f.write(f"Evaluation Results (Samples: {total}):\n")
            f.write(f"Answer Accuracy: {accuracy:.4f}\n")
            f.write(f"Format Accuracy: {format_accuracy:.4f}\n")
            f.write(f"Other Metrics: {average_metrics}\n")
            f.write(f"Total Retries: {total_retries}\n")
            f.write(f"Average Retries per Sample: {avg_retries:.2f}\n")

        return accuracy, format_accuracy, average_metrics

    def evaluate_model_parallel(self, generation_params=None,
                                seed=42, log_file="evaluation_log.txt", version=None,
                                max_parallel_gpus=None) -> Tuple[float, float]:
        """Parallel evaluation across multiple GPUs"""
        print("Starting parallel model evaluation...")
        random.seed(seed)
        # Detect available GPUs
        available_gpus = self.detect_available_gpus()

        if not available_gpus:
            print("No GPUs available, falling back to single-threaded CPU evaluation")
            return self.evaluate_model(generation_params, seed, log_file, version)

        # Determine number of parallel processes
        if max_parallel_gpus is not None:
            num_processes = min(max_parallel_gpus, len(available_gpus))
            available_gpus = available_gpus[:num_processes]
        else:
            num_processes = len(available_gpus)

        print(f"Using {num_processes} GPUs for parallel evaluation: {available_gpus}")

        # Load dataset
        eval_QAs = self.eval_dataset
        print(f"Evaluating {len(eval_QAs)} samples across {num_processes} GPUs")

        # Partition data
        partitions = self.partition_data(eval_QAs, num_processes)

        # Set up temporary files for results
        temp_dir = os.environ.get("TEMP_PARALLEL_FILE", "temp_parallel_results")
        os.makedirs(temp_dir, exist_ok=True)
        temp_files = [os.path.join(temp_dir, f"gpu_{gpu_id}_results.json") for gpu_id in available_gpus]

        # Store model path for subprocesses
        if hasattr(self, 'model_path'):
            os.environ["model_path"] = self.model_path
        else:
            print("Warning: model_path not found, make sure to call load_model first")

        # Start parallel processes
        print("Starting parallel evaluation processes...")
        processes = []

        # Use multiprocessing to run evaluation on each GPU
        mp.set_start_method('spawn', force=True)

        for i, gpu_id in enumerate(available_gpus):
            args = (gpu_id, partitions[i], generation_params, seed, log_file, temp_files[i])
            p = mp.Process(target=self._run_evaluation_process, args=args)  # 每个gpu跑一个process
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect and merge results
        print("Collecting results from all GPUs...")
        return self._collect_and_merge_results(temp_files, log_file, temp_dir)

    def evaluate_model_parallel_off(self, generation_params=None,
                                    seed=42, log_file="evaluation_log.txt", version=None,
                                    max_parallel_gpus=None) -> Tuple[float, float]:
        """Parallel evaluation across multiple GPUs"""
        print("Starting parallel model evaluation...")
        random.seed(seed)
        # Detect available GPUs
        available_gpus = self.detect_available_gpus()

        if not available_gpus:
            print("No GPUs available, falling back to single-threaded CPU evaluation")
            return self.evaluate_model_off(generation_params, seed, log_file, version)

        # Determine number of parallel processes
        if max_parallel_gpus is not None:
            num_processes = min(max_parallel_gpus, len(available_gpus))
            available_gpus = available_gpus[:num_processes]
        else:
            num_processes = len(available_gpus)

        print(f"Using {num_processes} GPUs for parallel evaluation: {available_gpus}")

        # Load dataset
        eval_QAs = self.eval_dataset
        print(f"Evaluating {len(eval_QAs)} samples across {num_processes} GPUs")

        # Partition data
        partitions = self.partition_data(eval_QAs, num_processes)

        # Set up temporary files for results
        temp_dir = os.environ.get("TEMP_PARALLEL_FILE", "temp_parallel_results")
        os.makedirs(temp_dir, exist_ok=True)
        temp_files = [os.path.join(temp_dir, f"gpu_{gpu_id}_results.json") for gpu_id in available_gpus]

        # Store model path for subprocesses
        if hasattr(self, 'model_path'):
            os.environ["model_path"] = self.model_path
        else:
            print("Warning: model_path not found, make sure to call load_model first")

        # Start parallel processes
        print("Starting parallel evaluation processes...")
        processes = []

        # Use multiprocessing to run evaluation on each GPU
        mp.set_start_method('spawn', force=True)

        for i, gpu_id in enumerate(available_gpus):
            args = (gpu_id, partitions[i], generation_params, seed, log_file, temp_files[i])
            p = mp.Process(target=self._run_evaluation_process_off, args=args)  # 每个gpu跑一个process
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect and merge results
        print("Collecting results from all GPUs...")
        return self._collect_and_merge_results(temp_files, log_file, temp_dir)

    def evaluate_partition(self, gpu_id: int, partition_data: List[Dict], generation_params: Dict,
                           seed: int, log_file: str, temp_results_file: str) -> Dict[str, Any]:
        """Evaluate a partition of data on a specific GPU"""
        try:
            # Set device for this process
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)

            print(f"Process {gpu_id}: Starting evaluation on {device} with {len(partition_data)} samples")

            # Load model on this GPU
            self.load_model(os.environ.get("model_path"), device)
            self.model.eval()
            random.seed(seed)

            correct = 0
            format_correct = 0
            other_metrics = []

            total = len(partition_data)
            total_retries = 0
            results = []

            for i, qa in enumerate(partition_data):
                self.model.reset_entropy_detection()
                self.model.reset_model_parameters()

                global_idx = qa['global_idx']

                if (i + 1) % 5 == 0:
                    print(f"GPU {gpu_id}: Evaluated {i + 1}/{total} samples")

                sys_prompt, demonstrations, prompt, syn_pair = qa['system'], qa['retrieved_texts'], qa['prompt'], qa[
                    'syn_pair']  # "", [(q,a),(q,a)],"" (完整cur prompt) ,"", (q,a); retrieved_texts就是demonstrations

                # semantics & second structure -> decorator
                prompt_text = self.build_prompt_text(sys_prompt, prompt)  # for semantics

                # structure list
                sys_prompt_inverse = self.tem_config['inv_sys']
                structure_text_lis = self.build_structure_text(sys_prompt, sys_prompt_inverse, demonstrations, syn_pair,
                                                               inverse=False)  # for structure

                inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True).to(
                    self.model.device)

                context = self.extract_contexts(prompt_text, strip_whitespace=False)
                context_len = self.tokenizer(context, return_tensors="pt", add_special_tokens=False,
                                             truncation=True).to(
                    self.model.device)
                context_len = context_len['input_ids'].shape[1]

                con_query_mask = torch.zeros_like(inputs['input_ids']).to(self.model.device)
                con_query_mask[:, context_len:] = 1

                struc_dataset = self.format_and_tokenize_dataset(structure_text_lis,
                                                                 target_device=self.model.device)
                self.model.struc_dataset, self.model.con_query_mask = struc_dataset, con_query_mask
                self.model.ttl_state = self.ttl_state
                use_entropy_control = os.environ.get("use_entropy_control", "False") == "True"
                if use_entropy_control:
                    max_retries = int(os.environ.get("max_retries", "5"))
                    completion, retry_count = self.generate_with_entropy_control(inputs, generation_params, max_retries,
                                                                                 context_len)
                else:
                    self.model.prompt_only = True
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_params,
                    )
                    completion = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                                       skip_special_tokens=True)
                    retry_count = 0

                # 使用后清理
                self.model.struc_dataset = None
                self.model.con_query_mask = None

                format_score = self.reward_format(qa, completion)
                correct_score = self.reward_correct(qa, completion)
                other_metrics_scores = self.reward_other_metrics(qa, completion)

                is_format_correct = format_score > 0
                is_answer_correct = correct_score > 0
                other_metrics.append(other_metrics_scores)

                if is_format_correct:
                    format_correct += 1
                if is_answer_correct:
                    correct += 1

                total_retries += retry_count

                # Store result for later sorting and logging
                result = {
                    'global_idx': global_idx,
                    'gpu_id': gpu_id,
                    'question': qa['Q'],
                    'model_response': completion,
                    'correct_answer': qa['A'],
                    'is_format_correct': is_format_correct,
                    'is_answer_correct': is_answer_correct,
                    'other_metrics_scores': other_metrics_scores,
                    'retry_count': retry_count
                }
                results.append(result)

                print(
                    f"GPU {gpu_id} - Sample {global_idx + 1}: Format={is_format_correct}, Answer={is_answer_correct}, Other Metric Score={other_metrics_scores}, Retries={retry_count}")

            # Save temporary results
            average_metrics = {k: round(sum(d[k] for d in other_metrics) / len(other_metrics), 4) for k in
                               (other_metrics[0].keys() if other_metrics else [])}

            temp_result = {
                'gpu_id': gpu_id,
                'correct': correct,
                'format_correct': format_correct,
                'other_metrics': average_metrics,
                'total': total,
                'total_retries': total_retries,
                'results': results
            }

            with open(temp_results_file, 'w') as f:
                json.dump(temp_result, f)

            print(
                f"GPU {gpu_id}: Completed evaluation. Accuracy: {correct / total:.4f}, Format: {format_correct / total:.4f}, Other Metrics: {average_metrics}")
            return temp_result

        except Exception as e:
            print(f"Error in GPU {gpu_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'gpu_id': gpu_id,
                'correct': 0,
                'format_correct': 0,
                'other_metrics': [],
                'total': 0,
                'total_retries': 0,
                'results': [],
                'error': str(e)
            }

    def evaluate_partition_off(self, gpu_id: int, partition_data: List[Dict], generation_params: Dict,
                               seed: int, log_file: str, temp_results_file: str) -> Dict[str, Any]:
        """Evaluate a partition of data on a specific GPU"""
        partition_data = partition_data[:2]  # 拿一个进行测试
        try:
            # Set device for this process
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)

            print(f"Process {gpu_id}: Starting evaluation on {device} with {len(partition_data)} samples")

            # Load model on this GPU
            self.load_model(os.environ.get("model_path"), device)
            self.model.eval()
            random.seed(seed)

            correct = 0
            format_correct = 0
            other_metrics = []
            total = len(partition_data)
            total_retries = 0
            results = []
            self.model.reset_offline_parameters()  # 设置为离线模式的参数
            for i, qa in enumerate(partition_data):
                self.model.reset_entropy_detection()
                self.model.reset_model_parameters()

                global_idx = qa['global_idx']

                if (i + 1) % 5 == 0:
                    print(f"GPU {gpu_id}: Evaluated {i + 1}/{total} samples")

                sys_prompt, demonstrations, prompt, syn_pair = qa['system'], qa['retrieved_texts'], qa['prompt'], qa[
                    'syn_pair']
                # semantics & second structure -> decorator
                prompt_text = self.build_prompt_text(sys_prompt, prompt)  # for semantics

                # structure list
                sys_prompt_inverse = self.tem_config['inv_sys']
                structure_text_lis = self.build_structure_text(sys_prompt, sys_prompt_inverse, demonstrations, syn_pair,
                                                               inverse=False)  # for structure

                inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True).to(
                    self.model.device)

                context = self.extract_contexts(prompt_text, strip_whitespace=False)
                context_len = self.tokenizer(context, return_tensors="pt", add_special_tokens=False,
                                             truncation=True).to(
                    self.model.device)
                context_len = context_len['input_ids'].shape[1]  # 一条条处理的

                con_query_mask = torch.zeros_like(inputs['input_ids']).to(self.model.device)  # 默认全0，不mask任何token
                con_query_mask[:, context_len:] = 1
                struc_dataset = self.format_and_tokenize_dataset(structure_text_lis,
                                                                 target_device=self.model.device)
                self.model.struc_dataset, self.model.con_query_mask = struc_dataset, con_query_mask
                self.model.ttl_state = self.ttl_state

                use_entropy_control = os.environ.get("use_entropy_control", "False") == "True"
                if use_entropy_control:
                    max_retries = int(os.environ.get("max_retries", "5"))
                    self.generate_with_entropy_control(inputs, generation_params, max_retries, context_len)
                else:
                    self.model.prompt_only = True
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_params,
                    )

                self.model.struc_dataset = None
                self.model.con_query_mask = None

            for i, qa in enumerate(partition_data):
                self.model.reset_entropy_detection()
                self.model.reset_model_parameters()

                global_idx = qa['global_idx']

                if (i + 1) % 5 == 0:
                    print(f"GPU {gpu_id}: Evaluated {i + 1}/{total} samples")

                sys_prompt, demonstrations, prompt, syn_pair = qa['system'], qa['retrieved_texts'], qa['prompt'], \
                    qa['syn_pair'] 

                # semantics & second structure -> decorator
                prompt_text = self.build_prompt_text(sys_prompt, prompt)  # for semantics

                # structure list
                sys_prompt_inverse = self.tem_config['inv_sys']
                structure_text_lis = self.build_structure_text(sys_prompt, sys_prompt_inverse, demonstrations,
                                                               syn_pair,
                                                               inverse=False)  # for structure

                inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True).to(
                    self.model.device)

                context = self.extract_contexts(prompt_text, strip_whitespace=False)
                context_len = self.tokenizer(context, return_tensors="pt", add_special_tokens=False,
                                             truncation=True).to(
                    self.model.device)
                context_len = context_len['input_ids'].shape[1]

                con_query_mask = torch.zeros_like(inputs['input_ids']).to(self.model.device)
                con_query_mask[:, context_len:] = 1
                struc_dataset = self.format_and_tokenize_dataset(structure_text_lis,
                                                                 target_device=self.model.device)
                self.model.struc_dataset, self.model.con_query_mask = struc_dataset, con_query_mask
                self.model.ttl_state = self.ttl_state

                use_entropy_control = os.environ.get("use_entropy_control", "False") == "True"
                if use_entropy_control:
                    max_retries = int(os.environ.get("max_retries", "5"))
                    completion, retry_count = self.generate_with_entropy_control_off(inputs, generation_params,
                                                                                     max_retries, context_len)
                else:
                    self.model.prompt_only = False
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_params,
                    )
                    completion = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                                       skip_special_tokens=True)
                    retry_count = 0

                # 使用后清理
                self.model.struc_dataset = None
                self.model.con_query_mask = None

                format_score = self.reward_format(qa, completion)
                correct_score = self.reward_correct(qa, completion)
                other_metrics_scores = self.reward_other_metrics(qa, completion)

                is_format_correct = format_score > 0
                is_answer_correct = correct_score > 0
                other_metrics.append(other_metrics_scores)  # [{'A':, 'B':}]

                if is_format_correct:
                    format_correct += 1
                if is_answer_correct:
                    correct += 1

                total_retries += retry_count

                # Store result for later sorting and logging
                result = {
                    'global_idx': global_idx,
                    'gpu_id': gpu_id,
                    'question': qa['Q'],
                    'model_response': completion,
                    'correct_answer': qa['A'],
                    'is_format_correct': is_format_correct,
                    'is_answer_correct': is_answer_correct,
                    'other_metrics_scores': other_metrics_scores,
                    'retry_count': retry_count
                }
                results.append(result)

                print(
                    f"GPU {gpu_id} - Sample {global_idx + 1}: Format={is_format_correct}, Answer={is_answer_correct}, Other Metric Score={other_metrics_scores}, Retries={retry_count}")

            # Save temporary results
            average_metrics = {k: round(sum(d[k] for d in other_metrics) / len(other_metrics), 4) for k in
                               (other_metrics[0].keys() if other_metrics else [])}

            temp_result = {
                'gpu_id': gpu_id,
                'correct': correct,
                'format_correct': format_correct,
                'other_metrics': average_metrics,
                'total': total,
                'total_retries': total_retries,
                'results': results
            }

            with open(temp_results_file, 'w') as f:
                json.dump(temp_result, f)
            print(
                f"GPU {gpu_id}: Completed evaluation. Accuracy: {correct / total:.4f}, Format: {format_correct / total:.4f}, Other Metrics: {average_metrics}")

            return temp_result

        except Exception as e:
            print(f"Error in GPU {gpu_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'gpu_id': gpu_id,
                'correct': 0,
                'format_correct': 0,
                'other_metrics': [],
                'total': 0,
                'total_retries': 0,
                'results': [],
                'error': str(e)
            }

    def _run_evaluation_process(self, gpu_id: int, partition_data: List[Dict],
                                generation_params: Dict, seed: int, log_file: str, temp_results_file: str):
        """Wrapper method to run evaluation in a separate process"""
        try:
            # Create a new evaluator instance for this process
            evaluator = self.__class__()
            evaluator.model_name = self.model_name
            evaluator.tem_config = self.tem_config
            evaluator.root = self.root
            evaluator.eval_dataset = self.eval_dataset
            evaluator.ttl_state = self.ttl_state
            result = evaluator.evaluate_partition(gpu_id, partition_data, generation_params,
                                                  seed, log_file, temp_results_file)
        except Exception as e:
            print(f"Process error on GPU {gpu_id}: {str(e)}")
            import traceback
            traceback.print_exc()

    def _run_evaluation_process_off(self, gpu_id: int, partition_data: List[Dict],
                                    generation_params: Dict, seed: int, log_file: str, temp_results_file: str):
        """Wrapper method to run evaluation in a separate process"""
        try:
            # Create a new evaluator instance for this process
            evaluator = self.__class__()
            evaluator.tem_config = self.tem_config  # 这里是新的，所以要注意传入对口的
            evaluator.root = self.root
            evaluator.ttl_state = self.ttl_state
            # evaluator.eval_dataset = self.eval_dataset
            result = evaluator.evaluate_partition_off(gpu_id, partition_data, generation_params,
                                                      seed, log_file, temp_results_file)
        except Exception as e:
            print(f"Process error on GPU {gpu_id}: {str(e)}")
            import traceback
            traceback.print_exc()

    def _collect_and_merge_results(self, temp_files: List[str], log_file: str, temp_dir: str) -> Tuple[float, float]:
        """Collect results from all GPUs and write ordered logs"""
        all_results = []
        total_correct = 0
        total_format_correct = 0
        total_other_metrics = []

        total_samples = 0
        total_retries = 0

        # Load all temporary results
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r') as f:
                        result = json.load(f)

                    if 'error' in result:
                        print(f"GPU {result['gpu_id']} had error: {result['error']}")
                        continue

                    total_correct += result['correct']
                    total_format_correct += result['format_correct']
                    total_other_metrics.append(result['other_metrics'])
                    total_samples += result['total']
                    total_retries += result['total_retries']

                    all_results.extend(result['results'])

                except Exception as e:
                    print(f"Error loading result file {temp_file}: {str(e)}")
            else:
                print(f"Warning: Result file {temp_file} not found")

        # Sort results by global index to maintain order
        all_results.sort(key=lambda x: x['global_idx'])

        # Write ordered log file
        with open(log_file, "a") as f:
            f.write(f"Number of evaluation samples: {total_samples}\n")
            f.write(f"Parallel evaluation across {len(temp_files)} GPUs\n\n")

            for result in all_results:
                f.write(f"Sample {result['global_idx'] + 1}:\n")
                f.write(f"GPU: {result['gpu_id']}\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Model Response: {result['model_response']}\n")
                f.write(f"Correct Answer: {result['correct_answer']}\n")
                f.write(
                    f"Format Correct: {result['is_format_correct']}, Answer Correct: {result['is_answer_correct']}\n")
                f.write(f"Other Metric Score: {result['other_metrics_scores']}\n")
                f.write(f"Retry Count: {result['retry_count']}\n\n")

        # Calculate final metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        format_accuracy = total_format_correct / total_samples if total_samples > 0 else 0
        avg_retries = total_retries / total_samples if total_samples > 0 else 0
        average_metrics = {k: round(sum(d[k] for d in total_other_metrics) / len(total_other_metrics), 4) for k in
                           (total_other_metrics[0].keys() if total_other_metrics else [])}

        print(f"\nParallel Evaluation Results (Samples: {total_samples}):")
        print(f"Answer Accuracy: {accuracy:.4f}")
        print(f"Format Accuracy: {format_accuracy:.4f}")
        print(f"Other Metrics: {average_metrics}")
        print(f"Total Retries: {total_retries}")
        print(f"Average Retries per Sample: {avg_retries:.2f}")

        # Write summary to log
        with open(log_file, "a") as f:
            f.write(f"Parallel Evaluation Results (Samples: {total_samples}):\n")
            f.write(f"Answer Accuracy: {accuracy:.4f}\n")
            f.write(f"Format Accuracy: {format_accuracy:.4f}\n")
            f.write(f"Other Metrics: {average_metrics}\n")
            f.write(f"Total Retries: {total_retries}\n")
            f.write(f"Average Retries per Sample: {avg_retries:.2f}\n")

        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {str(e)}")

        return accuracy, format_accuracy, average_metrics

