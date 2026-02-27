import os
import subprocess
import re

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import wandb
import warnings

import nltk
nltk.download('punkt_tab',download_dir='./my_nltk_cache')
warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
    category=DeprecationWarning
)


from data import get_task_fn, get_dataset_fn, split_dataset, gather_dataset
from loader import get_loader_fn, special_loader
from train import create_pure_model, create_pretrain_model
from train import ttl_online_train, ttl_offline_train, purellm_evaluation
from config import unify_config
from prompts import template_config
from utils import get_dir, set_random_seed, save_pickle, load_pickle, get_dir_data, transfer_dataset_format


set_random_seed(unify_config['SEED'])

def print_params(dataset, params):
    print("="*50)
    print(f"{dataset}参数配置详情")
    print("="*50)

    # 基础参数
    print("\n【基础参数】")
    basic_keys = ['SEED', 'USE_CUDA', 'GPU', 'GET_COT']
    for key in basic_keys:
        print(f"  {key:<20}: {params[key]}")

    # 生成参数
    print("\n【生成参数 (Generation)】")
    gen_keys = ['TEMP', 'DO_SAMPLE', 'MAX_LEN', 'MAX_RETRY', 'AVG', 'ENTROPY_THRE', 'LONG_ENTROPY_THRE']
    for key in gen_keys:
        print(f"  {key:<20}: {params[key]}")

    # TTL优化参数
    print("\n【TTL优化参数 (TTL optimization)】")
    ttl_keys = ['ITER_NUM', 'LR', 'ENTROPY_WEIGHT', 'STRU_WEIGHT', 'QUERY_WEIGHT',
                'ENTROPY_WIN', 'ENTROPY_TOKEN_MIN', 'LONG_ENTROPY_WIN', 'LONG_ENTROPY_TOKEN_MIN',
                'MIN_STD', 'MIN_THRESHOLD']
    for key in ttl_keys:
        print(f"  {key:<20}: {params[key]}")

    # 预训练参数
    print("\n【预训练参数 (Pretrain/SFT)】")
    pretrain_keys = ['PRETRAIN', 'PRETRAIN_OP', 'PRETRAIN_LORA_RANK', 'PRETRAIN_LORA_ALPHA',
                     'PRETRAIN_LORA_DROP', 'PRETRAIN_EPOCHS', 'PRETRAIN_BATCH_SIZE', 'GRADIENT_BACK',
                     'PRETRAIN_LR', 'PRETRAIN_WD', 'PRETRAIN_GEN_NUM', 'MAX_SFT_LEN', 'MAX_GEN_LEN']
    for key in pretrain_keys:
        print(f"  {key:<20}: {params[key]}")

    # 特殊参数
    print("\n【特殊参数 (Special)】")
    special_keys = ['TOPK']
    for key in special_keys:
        print(f"  {key:<20}: {params[key]}")

    print("\n" + "="*50)



def clear_gpu(gpu_id=2):
    """
    清空指定显卡的所有占用进程
    :param gpu_id: 显卡编号，默认清理第二个显卡（GPU 1）
    """
    try:
        # 1. 执行nvidia-smi命令，获取GPU占用信息
        result = subprocess.check_output(
            ['nvidia-smi', f'--query-compute-apps=pid,gpu_uuid,gpu_name', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )

        # 2. 解析输出，筛选出占用指定GPU的PID
        pid_list = []
        # 先获取指定GPU的UUID（避免显卡名称重复导致误杀）
        gpu_uuid_result = subprocess.check_output(
            ['nvidia-smi', f'--query-gpu=uuid', f'--id={gpu_id}', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        target_uuid = gpu_uuid_result.strip()

        # 遍历进程信息，筛选目标GPU的PID
        for line in result.strip().split('\n'):
            if not line:
                continue
            pid, uuid, _ = line.strip().split(', ')
            if uuid == target_uuid:
                pid_list.append(pid)

        # 3. 终止占用进程
        if pid_list:
            print(f"清理GPU {gpu_id} 上的进程，PID列表: {pid_list}")
            for pid in pid_list:
                try:
                    os.kill(int(pid), 9)  # 9表示强制终止进程
                    print(f"成功终止PID {pid} 的进程")
                except ProcessLookupError:
                    print(f"PID {pid} 的进程已不存在，跳过")
                except Exception as e:
                    print(f"终止PID {pid} 失败: {e}")
        else:
            print(f"GPU {gpu_id} 暂无占用进程，无需清理")

    except subprocess.CalledProcessError as e:
        print(f"执行nvidia-smi命令失败: {e}")
    except Exception as e:
        print(f"清理GPU {gpu_id} 失败: {e}")



def run_single_config(set_config, exp_num='0', retrain=False):
    """
    our main function
    :param set_config:
    :param exp_num:
    :return:
    """
    # config, task，output file settings
    pretrain, task_fn, task_name = get_task_fn(set_config)
    data_name, exp_num = set_config['DATASET'], exp_num
    root = set_config['ROOT']
    llm_name, ret_name = set_config['LLM'], set_config['EMB']

    print("retrain {} pretrain {} dataset {}; task mode {}; exp_num {}; root {}. ".format(retrain, pretrain, data_name, task_name, exp_num, root))
    adapter_root = get_dir(root, 'ckps/' + 'adapter_ckps/' + data_name + '-' + task_name + '-' + llm_name + '-' + exp_num)
    output_root = get_dir(root, 'ckps/' + 'output_ckps/' + data_name + '-' + task_name + '-' + llm_name+ '-' + exp_num)
    ttl_root = get_dir(root, 'ckps/' + 'ttl_ckps/' + data_name + '-' + task_name + '-' + llm_name+ '-' + exp_num)
    data_root = get_dir_data(root, data_name) # 数据存放位置
    aux_data_root = get_dir(root, 'datas/ready') # 辅助数据存放位置
    index_root = get_dir(root, 'index/' + data_name + '-' + task_name + '-' + ret_name + '-'+ exp_num + '/' + data_name) # index存放位置
    special_op = set_config['OP']  # in context / rag

    # dataset
    if os.path.exists(data_root + f'/train_dataset_{task_name}.pkl') and os.path.exists(data_root + f'/test_dataset_{task_name}.pkl'):
        print("Loading processed dataset from disk...")
        train_dic = load_pickle(data_root + f'/train_dataset_{task_name}.pkl')
        test_dic = load_pickle(data_root + f'/test_dataset_{task_name}.pkl')
    else:
        print("Processing dataset and saving to disk...")
        dataset = get_dataset_fn(data_name, data_root)
        train_dataset, test_dataset, data_mode = split_dataset(dataset, data_name) # 进行划分, 是否数据集有train_set

        train_dic, test_dic = gather_dataset(data_name, data_mode, task_fn, train_dataset, test_dataset, index_root, special_op, set_config)
        save_pickle(train_dic, data_root + f'/train_dataset_{task_name}.pkl')
        save_pickle(test_dic, data_root + f'/test_dataset_{task_name}.pkl')


    # loader, 好像不需要loader
    templates = template_config[data_name + '_' + task_name]
    train_dataset, test_dataset = get_loader_fn(data_root,task_name, train_dic, test_dic, templates, set_config, index_root=index_root, special_op=special_op)
    print("PHASE1: Data Done!")

    # pretrain(sft) or not
    if pretrain:
        print("PHASE2: Pretrain!")
        log_file, our_model, args, generation_params = create_pretrain_model(train_dataset, set_config, adapter_root, output_root, ttl_root, templates,retrain, benchmark_name=data_name + '-' + task_name + '-' + exp_num)
        print("PHASE2: Pretrain Done!")
    else:
        print("PHASE2: No Pretrain!")
        set_config['MODEL_PATH'] = set_config['LLM_PATH'] # 直接使用llm路径
        log_file, our_model, args, generation_params = create_pure_model(set_config, ttl_root, templates, benchmark_name=data_name + '-' + task_name + '-' + exp_num)

    # ttls
    ttl_state = set_config['TTL_STATE'] # 标明是否是online train or offline train
    if ttl_state == 'online':
        print("PHASE3: ON TTL!")
        ttl_online_train(log_file, our_model, args, generation_params, test_dataset)
    else:
        print("PHASE3: OFF TTL!")
        ttl_offline_train(log_file, our_model, args, generation_params, test_dataset)

    # save operation & other operation
    print("All Training Done!")







if __name__ == '__main__':
    print("Hi, TTL Health")
    # necessary config
    config = unify_config
    if config['DATASET'] in ['DiagnosisArena', 'ReDis', 'CupCase', 'MediQ', 'PubHealth']:
        print('Please make sure that the Diag task dataset is used in FREE/MCQ mode!')
    model_name = config['MODEL']
    config['PRETRAIN'] = False

    retrain = False
    exp_num = '3'
    config['ENSEMBLE'] = False
    config['GET_COT'] = False
    config['RE_RET'] = True # 重生成retrieval / 其他操作


    # without_train
    if config['DATASET'] in ["AIME2024", "AIME2025", 'MATH500', 'GPQA', 'HMMT2025', 'CupCase']:
        print("Dataset {} does not have training set, cannot run.".format(config['DATASET']))
        print("If you want to use them, please reset the topk=0 and do not use structure learning or change the structure learning into mask task")
        exit(0)

     for dataset in ['MedQA']:
        # 这两个一般钉死为5和0.1
        config['ITER_NUM'] = 5
        config['LR'] = 0.1
        config['ENTROPY_WEIGHT'] = 0.1
        tasks = ['MCQ']

        print_params(dataset, config)


        for task in tasks:
            config['TASK'] = task
            print("##################### Running dataset: {}, task: {} ########################".format(dataset, task))

            config['DATASET'] = dataset
            if model_name in ['ours']:
                run_single_config(config, retrain=retrain, exp_num=exp_num)
            else:
                print("No such model!")
