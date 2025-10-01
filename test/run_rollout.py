import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import multiprocessing as mp
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

# vllm is only required in non-API mode
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None
    logging.warning("vLLM is not installed. Please install vLLM if you need to use local model generation.")

from prompt_utils.utils_4 import call_api_repeat
def call_api_repeat_wrapper(prompt: str, api_model: str, api_platform: str, check_generate_time: str = "no", brace_type = '{',prepare_for_json = False):
    args = argparse.Namespace(
        model=api_model,
        platform=api_platform,
        check_generate_time=check_generate_time
    )
    return call_api_repeat(prompt, args, brace_type,prepare_for_json)

# --- Configure logging ---
logger = logging.getLogger(__name__)

def setup_logger():
    """Configures the global logger."""
    log_format = "[%(asctime)s - %(name)s - %(processName)s - %(levelname)s] - %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)

# --- Argument Parsing (same as previous version) ---
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Efficiently generate text using vLLM or an API.")
    
    # --- General Parameters ---
    parser.add_argument("--data_path", type=str, help="Data directory containing input JSON files.")
    parser.add_argument("--result_folder", type=str, default="results_llm_v0709", help="Directory to save output files.")
    parser.add_argument("--model_brief_name", type=str, default="model_results")
    parser.add_argument("--num_rollouts", type=int, default=1, help="Number of generations per prompt.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_num", type=int, default=-1, help="Maximum number of data entries to process per file, -1 for all.")
    parser.add_argument("--resume", action="store_true", help="Resume generation from where it left off.")

    # --- Prompt Construction Parameters ---
    parser.add_argument("--add_cot", action="store_true", help="Add 'Let's think step by step...' to the prompt.")
    parser.add_argument("--add_think_sys", action="store_true", help="Add guidance for the <think> tag in the prompt.")
    parser.add_argument("--no_system_prompt", action="store_true", help="Forcefully disable the system prompt")

    # --- Mode Selection ---
    parser.add_argument("--use_api", action="store_true", help="Use API for generation instead of local vLLM.")

    # --- vLLM Local Generation Parameters ---
    vllm_group = parser.add_argument_group('vLLM Local Generation Parameters')
    vllm_group.add_argument("--model_name", type=str, default="/home/zhoujinfeng/VERL_MODEL/cogreasoning/0614_sft_cog_data_v6_3_checkpoint_916/0614_sft_cog_data_v6_3_checkpoint_916", help="Path to the local model.")
    vllm_group.add_argument("--tokenizer_name", type=str, default="/home/zhoujinfeng/VERL_MODEL/cogreasoning/0614_sft_cog_data_v6_3_checkpoint_916/0614_sft_cog_data_v6_3_checkpoint_916", help="Path to the local tokenizer.")
    vllm_group.add_argument("--temperature", type=float, default=1)
    vllm_group.add_argument("--top_p", type=float, default=0.9)
    vllm_group.add_argument("--top_k", type=int, default=50)
    vllm_group.add_argument("--max_tokens", type=int, default=3000)
    vllm_group.add_argument("--batch_size", type=int, default=64, help="Batch size for each vLLM instance.")
    vllm_group.add_argument("--tp_per_instance", type=int, default=4, help="Tensor parallel size used by each vLLM instance (worker process).")
    vllm_group.add_argument("--gpu_devices", type=str, default=None, help='Specify GPU device IDs to use, comma-separated (e.g., "0,1,4,5"). If not specified, all available GPUs will be used.')

    # --- API Generation Parameters ---
    api_group = parser.add_argument_group('API Generation Parameters')
    api_group.add_argument("--api_model", type=str, default="qwen3_32B", help="Name of the API model to call.")
    api_group.add_argument("--api_platform", type=str, default="silicon_flow", help="Name of the API platform to call.")
    api_group.add_argument("--check_generate_time", type=str, default="no", choices=["yes", "no"], help="Whether to check API generation time.")
    api_group.add_argument("--num_api_workers", type=int, default=32, help="Number of concurrent processes to start when using API mode.")

    return parser.parse_args()


# --- Prompt Construction (vLLM specific) ---
def create_prompt(tokenizer: PreTrainedTokenizer, user_input: str, args: argparse.Namespace) -> str:
    system_prompt = "You are a helpful assistant."
    if args.add_think_sys:
        system_prompt += " You will always think before answer. Your thought should be wrapped in <think> and </think>. "

    prompt = user_input
    if args.add_cot:
        prompt += "\n\nLet's think step by step, and use <FINAL RESPONSE> before you give the final answer."

    if args.no_system_prompt:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    result = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert isinstance(result, str)
    return result

# --- Output Parsing ---
def parse_output(output_text: str, args: argparse.Namespace) -> Dict[str, Any]:
    if output_text.endswith('<|im_end|>'):
        output_text = output_text[:-len('<|im_end|>')]
    valid = True
    
    think, response = "", output_text
    if args.add_think_sys:
        parts = output_text.split('</think>')
        if len(parts) != 2: valid = False
        think, response = parts[0], parts[-1]
        
    response_cot, response_real = "", response
    if args.add_cot:
        parts = response.split('<FINAL RESPONSE>')
        if len(parts) != 2: valid = False
        response_cot, response_real = parts[0], parts[-1]
    
    return {'think': think.strip(), 'response_cot': response_cot.strip(), 'response_real': response_real.strip(), 'valid': valid}

# Parsing function designed for API return results, with added CoT support
def parse_api_output(response: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Parses the return result of the call_api_repeat function to align its format with parse_output and support CoT.
    """
    think_content = ""
    full_response = ""
    valid = True
    reasoning_tokens = 0
    output_tokens = 0

    if isinstance(response, dict):
        think_content = response.get("reasoning_content", "")
        full_response = response.get("content", "")
        reasoning_tokens = response.get("reasoning_tokens", 0)
        output_tokens = response.get("output_tokens", 0)
    elif isinstance(response, str):
        full_response = response
    else:
        full_response = f"Unknown API return type: {type(response)}"
        valid = False
    
    response_cot, response_real = "", full_response
    if args.add_cot:
        parts = full_response.split('<FINAL RESPONSE>')
        if len(parts) == 2:
            response_cot, response_real = parts[0], parts[-1]
        else:
            # Even if CoT is enabled, if the API does not return a separator, it is considered a real response
            valid = False 
    
    return {
        'think': think_content,
        'response_cot': response_cot,
        'response_real': response_real,
        'valid': valid,
        'reasoning_tokens': reasoning_tokens,
        'output_tokens': output_tokens
    }

# --- vLLM Worker Process ---
def run_worker(*args, **kwargs):
    worker_id, gpu_devices, prompts, metadata, sampling_params, sampling_mode, save_file_path, lock, args_ns = args
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    setup_logger()
    logger.info(f"Worker-{worker_id} started, using GPUs: [{gpu_devices}]")

    if LLM is None:
        raise ImportError("vLLM is not installed and cannot be run in local mode.")
        
    llm = LLM(model=args_ns.model_name, tokenizer=args_ns.tokenizer_name, trust_remote_code=False, dtype="half", tensor_parallel_size=args_ns.tp_per_instance)
    logger.info(f"Worker-{worker_id} model loaded.")
    
    pbar = tqdm(total=len(prompts), desc=f"Worker-{worker_id} generating", position=worker_id)
    for i in range(0, len(prompts), args_ns.batch_size):
        batch_prompts = prompts[i:i + args_ns.batch_size]
        batch_meta = metadata[i:i + args_ns.batch_size]
        
        batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        
        results_to_save = []
        tokenizer = llm.get_tokenizer()
        for output, meta in zip(batch_outputs, batch_meta):
            output_text = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=False)
            parsed_output = parse_output(output_text, args_ns)
            
            result = {**meta, 'sampling_mode': sampling_mode, 'llm_output': parsed_output}
            results_to_save.append(json.dumps(result, ensure_ascii=False) + "\n")

        with lock:
            with open(save_file_path, "a", encoding='utf-8') as f:
                f.writelines(results_to_save)
        pbar.update(len(batch_prompts))
    pbar.close()
    logger.info(f"Worker-{worker_id} finished processing.")


# Single task processing function designed for the process pool
g_lock = None
g_args = None
g_save_path = None

def init_pool_worker(lock, args_ns, save_path):
    """Initializes global variables for each worker process in the pool."""
    global g_lock, g_args, g_save_path
    g_lock = lock
    g_args = args_ns
    g_save_path = save_path

def process_single_api_prompt(task_data):
    """
    Function to process a single API request, to be called by the process pool.
    """
    prompt, meta = task_data
    try:
        # Call the API function
        api_response = call_api_repeat_wrapper(
            prompt=prompt,
            api_model=g_args.api_model,
            api_platform=g_args.api_platform,
            check_generate_time=g_args.check_generate_time, 
            prepare_for_json=False, 
        )
        
        # Parse the API response
        parsed_output = parse_api_output(api_response, g_args)

        result = {
            **meta,
            'sampling_mode': 'api',
            'llm_output': parsed_output
        }
        
        # Use the lock obtained from the initializer to write to the file safely
        with g_lock:
            with open(g_save_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        return True # Indicates success
    except Exception as e:
        logger.exception(f"Error processing idx={meta.get('idx')}: {e}")
        return False # Indicates failure


# --- Main Function (Coordinator) ---
def main():
    """Main execution function, responsible for coordinating and dispatching tasks."""
    setup_logger()
    args = parse_arguments()
    logger.info(f"Program started with arguments: {args}")

    random.seed(args.seed)

    suffix = "api" if args.use_api else ("greedy" if args.num_rollouts <= 1 else "random")
    result_file = Path(args.result_folder) / f"{args.model_brief_name}_results_{suffix}.jsonl"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    existing_counts = defaultdict(int)
    if result_file.exists() and args.resume:
        logger.info(f"Resuming from file {result_file}. Loading existing results...")
        with open(result_file, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    unique_id = f"{item['origin_data']['input']}{item['file_name']}"
                    existing_counts[unique_id] += 1
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"Skipping invalid line in results file: {line.strip()}")
        logger.info(f"Loaded {sum(existing_counts.values())} existing results.")
    elif result_file.exists() and not args.resume:
        logger.warning(f"Result file {result_file} exists but --resume is not set. This file will be overwritten.")
        os.remove(result_file)


    # Preload and prepare all data
    logger.info("Collecting and preparing prompts from all files...")
    main_tokenizer = None
    if not args.use_api:
        # Load tokenizer only in vLLM mode
        main_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=False)

    all_prompts, all_metadata = [], []
    data_path = Path(args.data_path)
    files_to_process = [f for f in data_path.iterdir() if f.is_file() and f.suffix == '.json'] if data_path.is_dir() else [data_path]
    
    for file_path in files_to_process:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        if args.data_num > 0:
            random.shuffle(data)
            data = data[:args.data_num]
        
        for i, d in enumerate(data):
            unique_id = f"{d['input']}{file_path.name}"
            if existing_counts.get(unique_id, 0) >= args.num_rollouts:
                continue
            
            # Prepare prompt based on the mode
            if args.use_api:
                # API mode: directly use user_input, append CoT instruction as needed
                prompt_str = d['input']
                if args.add_cot:
                    prompt_str += "\n\nLet's think step by step, and use <FINAL RESPONSE> before you give the final answer."
            else:
                # vLLM mode: use the original chat template
                prompt_str = create_prompt(main_tokenizer, d['input'], args)
            
            for k in range(existing_counts.get(unique_id, 0), args.num_rollouts):
                all_prompts.append(prompt_str)
                all_metadata.append({'idx': i, 'file_name': file_path.name, 'origin_data': d, 'rollout_idx': k})
    
    if not all_prompts:
        logger.info("All entries have been generated, no execution needed.")
        return

    # --- Task Dispatch and Execution ---
    if args.use_api:
        # API execution path: use a process pool
        num_workers = args.num_api_workers
        logger.info(f"Entering API mode. A total of {len(all_prompts)} new prompts have been prepared and will be processed using a pool of {num_workers} processes.")

        # Pack prompts and metadata into a task list
        tasks = list(zip(all_prompts, all_metadata))
        
        # Use a Manager to create a lock that can be shared between processes
        manager = mp.Manager()
        lock = manager.Lock()
        
        with mp.Pool(processes=num_workers, initializer=init_pool_worker, initargs=(lock, args, result_file)) as pool:
            # Use tqdm to display a progress bar
            with tqdm(total=len(tasks), desc="Executing API calls") as pbar:
                # imap_unordered can get results faster, suitable for I/O-intensive tasks
                for _ in pool.imap_unordered(process_single_api_prompt, tasks):
                    pbar.update()

    else:
        # --- vLLM Local Execution Path ---
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA environment not detected, vLLM requires GPU support.")
        
        target_gpu_ids = []
        all_available_gpu_ids = list(range(torch.cuda.device_count()))
        if args.gpu_devices:
            target_gpu_ids = [int(g.strip()) for g in args.gpu_devices.split(',')]
        else:
            target_gpu_ids = all_available_gpu_ids
            
        tp_size = args.tp_per_instance
        num_gpus_to_use = len(target_gpu_ids)
        if num_gpus_to_use < tp_size:
            raise ValueError(f"Number of GPUs ({num_gpus_to_use}) is less than TP size ({tp_size}).")
        
        if num_gpus_to_use % tp_size != 0:
            usable_gpu_count = (num_gpus_to_use // tp_size) * tp_size
            logger.warning(f"Total number of GPUs ({num_gpus_to_use}) is not divisible by TP size ({tp_size}). Only the first {usable_gpu_count} GPUs will be used.")
            target_gpu_ids = target_gpu_ids[:usable_gpu_count]

        num_workers = len(target_gpu_ids) // tp_size
        logger.info(f"Will use {len(target_gpu_ids)} GPUs {target_gpu_ids} to start {num_workers} vLLM worker processes.")
        
        sampling_params = None
        sampling_mode = ''
        if args.num_rollouts <= 1:
            sampling_mode = 'greedy'
            sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0, top_p=1.0, top_k=-1, skip_special_tokens=False)
        else:
            sampling_mode = 'random'
            sampling_params = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens, skip_special_tokens=False)
        
        processes = []
        prompts_per_worker = [all_prompts[i::num_workers] for i in range(num_workers)]
        metadata_per_worker = [all_metadata[i::num_workers] for i in range(num_workers)]
        file_lock = mp.Lock()

        for i in range(num_workers):
            if not prompts_per_worker[i]: continue
            
            start_index = i * tp_size
            end_index = start_index + tp_size
            worker_gpu_ids = target_gpu_ids[start_index:end_index]
            gpu_devices_str = ",".join(map(str, worker_gpu_ids))
            
            process = mp.Process(
                target=run_worker, name=f"Worker-{i}",
                args=(
                    i, gpu_devices_str, prompts_per_worker[i], metadata_per_worker[i],
                    sampling_params, sampling_mode, result_file, file_lock, args
                )
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    logger.info("All worker processes have finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
