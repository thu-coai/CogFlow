import json
import random
import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import DataParallel
from transformers import AutoTokenizer, AutoModelForTokenClassification, PreTrainedModel, PreTrainedTokenizer
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import multiprocessing
from pathlib import Path
import tiktoken



# --- Setup logging ---
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Dependencies and Configurations ---
try:
    from prompt_utils import eval_responses
except ImportError:
    eval_responses = None
    logger.warning("Could not import 'eval_responses' from 'prompt_utils.chain_output_eval'. PromptEvaluator will not be available.")

try:
    # FIX: Adhering strictly to the user's required import statement for their environment.
    from google import genai
except ImportError:
    genai = None
    logger.warning("Could not import 'genai' from 'google'. Please ensure the correct library (e.g., 'google-genai') is installed. Gemini token counting will be unavailable.")

try:
    from config_tokenizer import tokenizer_name_of_files
except ImportError:
    tokenizer_name_of_files = {}
    logger.warning("confg_tokenizer.py not found or is empty. Automatic tokenizer selection will be disabled.")


# =====================================================================================
#  Tokenizer Loader for Analysis
# =====================================================================================
class GeminiTokenizer:
    """A wrapper for the google.generativeai library to count tokens, using the Client pattern."""
    def __init__(self, model_name: str):
        if genai is None:
            raise ImportError("The 'google-genai' library is required for GeminiTokenizer.")
        self.model_name = model_name
        self.client = None
        try:
            # This requires the GOOGLE_API_KEY environment variable to be set.
            self.client = genai.Client()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client. Make sure your API key is configured as an environment variable (GOOGLE_API_KEY). Error: {e}")
            self.client = None

    def count_tokens(self, text: str) -> int:
        if not self.client or not text:
            return 0
        try:
            # Use the client.models.count_tokens method and use the model name directly.
            response = self.client.models.count_tokens(model=self.model_name, contents=[text])
            return response['total_tokens']
        except Exception as e:
            logger.warning(f"Gemini token count failed for model {self.model_name}. This may be due to an incompatible library version, invalid API key, or invalid model name. Error: {e}. Returning 0.")
            return 0

class GenericTokenizer:
    """A wrapper to provide a consistent interface for different tokenizers."""
    def __init__(self, tokenizer_impl):
        self.tokenizer = tokenizer_impl

    def count_tokens(self, text: str) -> int:
        if isinstance(self.tokenizer, PreTrainedTokenizer):  # HuggingFace
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        elif hasattr(self.tokenizer, 'encode'):  # tiktoken
            return len(self.tokenizer.encode(text))
        else:
            raise TypeError(f"Unsupported tokenizer implementation: {type(self.tokenizer)}")

def get_tokenizer_for_analysis(tokenizer_name_or_path: str):
    """
    Loads a tokenizer for analysis based on a name or path.
    Handles special cases like 'gemini-*', 'gpt-4o', 'glm-4.5'.
    """
    if not tokenizer_name_or_path:
        return None
    logger.info(f"Loading tokenizer for analysis: {tokenizer_name_or_path}")
    try:
        if tokenizer_name_or_path.startswith("gemini"):
            if genai:
                return GeminiTokenizer(model_name=tokenizer_name_or_path)
            else:
                logger.warning("Attempted to use Gemini tokenizer, but 'google.genai' is not installed.")
                return None
        elif tokenizer_name_or_path == "gpt-4o":
            encoding = tiktoken.get_encoding("o200k_base")
            return GenericTokenizer(encoding)
        elif tokenizer_name_or_path == "glm-4.5":
            return GenericTokenizer(AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True))
        else:
            return GenericTokenizer(AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True))
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{tokenizer_name_or_path}'. Token counting will be skipped. Error: {e}")
        return None


# =====================================================================================
#  Evaluator Class for Method 1: Model-Based Evaluation (Modified for Multi-GPU)
# =====================================================================================
class ModelEvaluator:
    """
    Evaluates responses using a pre-trained reward model.
    Modified to leverage DataParallel for multi-GPU acceleration.
    """
    rm_instruction_template = \
"""[Task]
Given a user query ([Input]), multiple reference responses ([Reference Responses]), and a candidate response for evaluation ([Candidate Response]).

The reference responses are given in order, and the first reference response is the best one. You should determine whether the candidate response strictly outperforms all reference responses. Thus, 0 means the candidate response is the best one, 1 means the candidate response is worse than at least one reference responses.

[Input]
{user_input}

[Reference Responses]
{reference_responses}

[Candidate Response]
{candidate_response}

[Output]
The rank of the candidate response is: """

    def __init__(self, args):
        if args.analysis_only:
            logger.info(f"Analysis-only mode, skipping evaluator loading.")
            return
        
        logger.info(f"Initializing ModelEvaluator with model: {args.model_name}")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        
        model = AutoModelForTokenClassification.from_pretrained(args.model_name)

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for Data Parallel evaluation.")
            self.rm = DataParallel(model)
        else:
            self.rm = model
        
        self.rm.to("cuda")
        self.rm.eval()

        self.batch_size = args.batch_size
        self.args = args

    def evaluate(self, data_to_process, save_file_path, save_file_brief_path):
        all_prompts = []
        metadata = []

        logger.info("Preparing prompts for model-based evaluation...")
        for d in tqdm(data_to_process, desc="Preparing Prompts"):
            user_input = d['origin_data']['input']
            reference_responses = [item['reasoning_and_response'] for item in d['origin_data']['reference_responses_from_diverse_source'] if item['source'] == 'cogflow']
            reference_responses = reference_responses[:3]
            reference_responses = [item.split('</think>')[-1].split('<|im_end|>')[0] for item in reference_responses]
            reference_responses = [{"rank": i + 1, "content": item} for i, item in enumerate(reference_responses)]
            
            for c in self.args.remove_char:
                user_input = user_input.replace(c, '')
                for ref in reference_responses:
                    ref['content'] = ref['content'].replace(c, '')
            
            prompt = self.rm_instruction_template.format(
                user_input=user_input,
                reference_responses=json.dumps(reference_responses, indent=4, ensure_ascii=False),
                candidate_response=d['llm_output']['response_real'],
            )
            
            messages = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            assert isinstance(prompt_str, str), f"prompt_str is not a string: {type(prompt_str)}"
            prompt_str = prompt_str.strip().removesuffix('<|im_end|>')

            all_prompts.append(prompt_str)
            metadata.append(d)

        total_batches = (len(all_prompts) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(all_prompts), self.batch_size), desc="Running Model Evaluation", total=total_batches):
            batch_prompts = all_prompts[i:i + self.batch_size]
            batch_meta = metadata[i:i + self.batch_size]
            
            model_config = self.rm.module.config if hasattr(self.rm, 'module') else self.rm.config
            
            encoded_inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, padding_side="left", truncation=True,
                max_length=model_config.n_positions if hasattr(model_config, 'n_positions') else 8192,
            ).to("cuda")
            
            with torch.no_grad():
                outputs = self.rm(**encoded_inputs, return_dict=True)

            logits = outputs.logits[:, -1, :]

            with open(save_file_path, "a", encoding='utf-8') as f_full, \
                 open(save_file_brief_path, "a", encoding='utf-8') as f_brief:
                for idx in range(logits.shape[0]):
                    d = batch_meta[idx]
                    output = logits[idx]
                    
                    softmax_scores = torch.softmax(output, dim=-1)
                    score = float(softmax_scores[0])

                    out = d.copy()
                    out['rm_output'] = {
                        'prompt': batch_prompts[idx],
                        'origin_output': output.cpu().numpy().tolist(),
                        'softmax_output': softmax_scores.cpu().numpy().tolist(),
                        'score': score
                    }
                    
                    f_full.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f_brief.write(json.dumps(out['rm_output'], ensure_ascii=False) + "\n")
        logger.info("Model-based evaluation finished.")


# =====================================================================================
#  Evaluator Class for Method 2: Prompt-Based (API) Evaluation
# =====================================================================================
class PromptEvaluator:
    """
    Evaluates responses using a prompt-based method, likely via an API call.
    """
    def __init__(self, args):
        logger.info("Initializing PromptEvaluator.")
        if eval_responses is None:
            raise RuntimeError("PromptEvaluator requires 'eval_responses' from 'prompt_utils', which is not available.")
        self.lock = multiprocessing.Lock()
        self.args = args

    def _do_eval_retry(self, user_input, cur_responses, reference_responses, score_type):
        for _ in range(3):
            try:
                all_score, all_ref_score, all_eval_score = eval_responses(user_input, cur_responses, reference_responses, self.args.prompt_eval_model_name, score_type, self.args.platform)
                return all_score, all_ref_score, all_eval_score
            except Exception as e:
                logger.error(f"Error in eval_responses, retrying... Error: {e}")
        return [-100], [], []

    def _eval_and_output(self, d, save_file_path, save_file_brief_path, score_type):
        if 'response_to_eval' in d:
            user_input = d['input']
            reference_responses = [item['reasoning_and_response'] for item in d['reference_responses_from_diverse_source'] if item['source'] == 'cogflow']
            reference_responses = [item.split('</think>')[-1] for item in reference_responses]
            cur_responses = d['response_to_eval']
        else:
            user_input = d['origin_data']['input']
            reference_responses = [item['reasoning_and_response'] for item in d['origin_data']['reference_responses_from_diverse_source'] if item['source'] == 'cogflow']
            reference_responses = [item.split('</think>')[-1] for item in reference_responses]
            cur_responses = [d['llm_output']['response_real']]
        
        all_score, all_ref_score, all_eval_score = self._do_eval_retry(user_input, cur_responses, reference_responses, score_type)
        
        out = d.copy()
        out['rm_output'] = {
            'all_score': all_score,
            'all_ref_score': all_ref_score,
            'all_eval_score': all_eval_score,
            'score': float(all_score[0]) if all_score and all_score[0] is not None else -100.0
        }
        
        with self.lock:
            with open(save_file_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            with open(save_file_brief_path, "a", encoding='utf-8') as f_brief:
                f_brief.write(json.dumps(out['rm_output'], ensure_ascii=False) + "\n")

    def evaluate(self, data_to_process, save_file_path, save_file_brief_path, score_type = "mid"):
        logger.info(f"Starting prompt-based evaluation for {len(data_to_process)} items...")
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            futures = [executor.submit(self._eval_and_output, d, save_file_path, save_file_brief_path, score_type) for d in data_to_process]
            for future in tqdm(futures, desc="Running Prompt Evaluation", total=len(data_to_process)):
                future.result()
        logger.info("Prompt-based evaluation finished.")


# =====================================================================================
#  Analysis Function
# =====================================================================================
def analyze_results(result_file_path: str, filter_option: str = "none"):
    """
    Loads an evaluation result file and calculates statistics.
    - Automatically determines the tokenizer based on the filename and confg_tokenizer.py.
    - If filter_option="none", calculates for all data and provides default values for missing depth/type.
    """
    if not os.path.exists(result_file_path):
        logger.error(f"Analysis failed. Result file not found: {result_file_path}")
        return

    tokenizer = None
    try:
        stem = Path(result_file_path).stem
        if "_on_" in stem and stem.startswith("eval_"):
            feature_name = stem.split("_on_")[0][len("eval_"):]
            if feature_name in tokenizer_name_of_files:
                tokenizer_name = tokenizer_name_of_files[feature_name]
                logger.info(f"Found tokenizer mapping for '{feature_name}': '{tokenizer_name}'")
                tokenizer = get_tokenizer_for_analysis(tokenizer_name)
            else:
                logger.warning(f"Tokenizer mapping for feature file '{feature_name}' not found in confg_tokenizer.py. Skipping token counting.")
        else:
            logger.warning(f"Filename '{stem}' does not match the 'eval_*_on_*' format. Cannot determine tokenizer.")
    except Exception as e:
        logger.error(f"Error selecting tokenizer: {e}. Skipping token counting.")

    logger.info(f"--- Starting analysis of {result_file_path} ---")
    
    grouped_results = defaultdict(list)
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'idx' in item and 'file_name' in item and 'rm_output' in item and 'score' in item['rm_output']:
                        group_key = (item['idx'], item['file_name'])
                        grouped_results[group_key].append(item)
                    else:
                        logger.warning(f"Skipping malformed item in result file: {item}")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line.strip()}")
    except FileNotFoundError:
        logger.error(f"Could not open result file: {result_file_path}")
        return

    if not grouped_results:
        logger.warning("No valid data found for analysis in the result file.")
        return

    node_list = ["Observation", "Attribution", "Motivation", "Regulation", "Efficacy", "Behavior"]

    def get_ave_depth(node_cnts: list): 
        depths = [sum(node_cnt.values()) for node_cnt in node_cnts[:3]]
        return np.mean(depths) if depths else 0
    
    def get_ave_type(node_cnts: list):
        hav_cnts = [sum(1 for t in node_list if node_cnt.get(t, 0) > 0) for node_cnt in node_cnts[:3]]
        return np.mean(hav_cnts) if hav_cnts else 0

    stats = defaultdict(lambda: defaultdict(list))
    
    for group_key, rollouts in grouped_results.items():
        if not rollouts: continue
        
        data_source = rollouts[0]['origin_data'].get('data_source', 'cogflow')
        user_input = rollouts[0]['origin_data']['input']
        curr_diff = rollouts[0]['origin_data']['difficulty']
        actual_data = rollouts[0]['origin_data']

        # Select different processing logic based on the filter parameter
        if filter_option != "none":
            
            # Difficulty filtering
            if (filter_option in ["easy", "medium", "hard"]) and (curr_diff != filter_option):
                continue

            # Calculate actual depth and type
            node_cnts = []
            if 'reference_responses_from_diverse_source' in actual_data:
                for ref_dict in actual_data.get('reference_responses_from_diverse_source', []):
                    if ref_dict.get('source') != 'fake_variations':
                        ref = ref_dict.get('reasoning_and_response', '')
                        node_cnts.append({node: ref.count(f"<{node}>") for node in node_list})
            
            stats[data_source]['depth'].append(get_ave_depth(node_cnts))
            stats[data_source]['type'].append(get_ave_type(node_cnts))
        
        else: # filter_option == "none"
            # Permissive mode: No filtering, process all data
            # Try to calculate depth and type, use default value 0 if data does not exist
            current_depth = 0.0
            current_type = 0.0
            node_cnts = []
            if 'reference_responses_from_diverse_source' in actual_data:
                for ref_dict in actual_data.get('reference_responses_from_diverse_source', []):
                    if ref_dict.get('source') != 'fake_variations':
                        ref = ref_dict.get('reasoning_and_response', '')
                        node_cnts.append({node: ref.count(f"<{node}>") for node in node_list})
            current_depth = get_ave_depth(node_cnts)
            current_type = get_ave_type(node_cnts)
            
            stats[data_source]['depth'].append(current_depth)
            stats[data_source]['type'].append(current_type)

        # --- The following statistics logic is common to both modes ---
        best_rollout = max(rollouts, key=lambda x: x['rm_output']['score'])
        if best_rollout['rm_output']['score'] >= 0:
            stats[data_source]['best_of_n_scores'].append(best_rollout['rm_output']['score'])
        
        for r in rollouts:
            if r['rm_output']['score'] >= 0:
                stats[data_source]['individual_scores'].append(r['rm_output']['score'])
                
                think_text = r.get('llm_output', {}).get('think', '') or ""
                think_text += r.get('llm_output', {}).get('response_cot', '') or ""
                response_text = r.get('llm_output', {}).get('response_real', '') or ""
                
                stats[data_source]['think_lengths'].append(len(think_text))
                stats[data_source]['response_lengths'].append(len(response_text))
                
                if tokenizer:
                    stats[data_source]['think_tokens'].append(tokenizer.count_tokens(think_text))
                    stats[data_source]['response_tokens'].append(tokenizer.count_tokens(response_text))

    logger.info("--- Analysis Results ---")
    
    for data_source in sorted(stats.keys()):
        ds_stats = stats[data_source]
        num_groups = len(ds_stats['best_of_n_scores'])
        num_rollouts = len(ds_stats['individual_scores'])
        
        if bon_scores := ds_stats['best_of_n_scores']:
            logger.info(f"[{data_source}][Best-of-N] ({num_groups} prompts): {np.mean(bon_scores):.4f} ± {np.std(bon_scores):.4f}")
        if overall_scores := ds_stats['individual_scores']:
            logger.info(f"[{data_source}][Overall] ({num_rollouts} rollouts): {np.mean(overall_scores):.4f} ± {np.std(overall_scores):.4f}")
        if t_lengths := ds_stats['think_lengths']:
            logger.info(f"[{data_source}][Stats] Average think length (chars): {np.mean(t_lengths):.2f}")
        if tokenizer and (t_tokens := ds_stats['think_tokens']):
            logger.info(f"[{data_source}][Stats] Average think length (tokens): {np.mean(t_tokens):.2f}")
        if r_lengths := ds_stats['response_lengths']:
            logger.info(f"[{data_source}][Stats] Average response length (chars): {np.mean(r_lengths):.2f}")
        if tokenizer and (r_tokens := ds_stats['response_tokens']):
            logger.info(f"[{data_source}][Stats] Average response length (tokens): {np.mean(r_tokens):.2f}")
        if depths := ds_stats.get('depth'):
            logger.info(f"[{data_source}][Stats] Average depth: {np.mean(depths):.2f}")
        if types := ds_stats.get('type'):
            logger.info(f"[{data_source}][Stats] Average type: {np.mean(types):.2f}")
    
    logger.info("--- Analysis Finished ---")


# =====================================================================================
#  Main Execution Logic
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script with Statistics")
    parser.add_argument("--llm_result", type=str, required=True, help="Path to the single .jsonl file generated by the LLM or a folder of .jsonl files.")
    parser.add_argument("--eval_result_folder", type=str, default="results_eval", help="Directory to save evaluation results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from previously saved evaluation results.")
    parser.add_argument("--eval_method", type=str, required=True, choices=['model', 'prompt'], help="The evaluation method to use.")
    parser.add_argument("--remove_char", type=str, default='*')
    
    # Analysis Arguments
    parser.add_argument("--perform_analysis", action="store_true", default=True, help="Automatically run analysis after evaluation.")
    parser.add_argument("--no_analysis", action="store_false", dest="perform_analysis")
    parser.add_argument("--analysis_only", action="store_true", help="Skip evaluation and only run analysis on an existing result file.")
    parser.add_argument("--analysis_filter", type=str, default="none", choices=['none', 'easy', 'medium', 'hard'], help="Filter data by difficulty during analysis.")

    # Method-specific Arguments
    parser.add_argument("--model_name", type=str,  help="Path to the reward model (for 'model' method).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for 'model' method.")
    parser.add_argument("--prompt_eval_model_name", type=str, help="Model name for prompt-based evaluator.")
    parser.add_argument("--platform", type=str, default="custom")
    parser.add_argument("--max_workers", type=int, default=32, help="Max concurrent workers for 'prompt' method.")
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.eval_method == 'model':
        if not args.model_name and not args.analysis_only:
            parser.error("--model_name is required when --eval_method is 'model'")
        model_name_for_path = Path(args.model_name).name if args.model_name else "default_model"
        evaluator = ModelEvaluator(args)
    elif args.eval_method == 'prompt':
        model_name_for_path = args.prompt_eval_model_name
        evaluator = PromptEvaluator(args)

    llm_result_files = [os.path.join(args.llm_result, f) for f in os.listdir(args.llm_result) if f.endswith('.jsonl')] if os.path.isdir(args.llm_result) else [args.llm_result]

    for llm_result_file in llm_result_files:
        os.makedirs(args.eval_result_folder, exist_ok=True)
        llm_result_basename = Path(llm_result_file).stem
        save_file_path = os.path.join(args.eval_result_folder, f"eval_{llm_result_basename}_on_{model_name_for_path}.jsonl")
        save_file_brief_path = os.path.join(args.eval_result_folder, f"eval_{llm_result_basename}_on_{model_name_for_path}_brief.jsonl")

        if args.analysis_only:
            analyze_results(save_file_path, args.analysis_filter)
            continue

        if not os.path.exists(llm_result_file):
            logger.error(f"LLM result file not found: {llm_result_file}")
            continue
        with open(llm_result_file, "r", encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
        
        existing_ids = set()
        if args.resume and os.path.exists(save_file_path):
            with open(save_file_path, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if item.get('rm_output',{}).get('score', -100) > -10:
                            existing_ids.add((item['origin_data']['input'], item['file_name'], item['rollout_idx']))
                    except (json.JSONDecodeError, KeyError): continue
            logger.info(f"Found {len(existing_ids)} existing evaluated results. Resuming.")
        elif os.path.exists(save_file_path):
            logger.warning(f"Result file {save_file_path} exists and --resume not set. Overwriting.")
            os.remove(save_file_path)
            if os.path.exists(save_file_brief_path): os.remove(save_file_brief_path)

        data_to_process = [d for d in all_data if (d['origin_data']['input'], d['file_name'], d['rollout_idx']) not in existing_ids]

        if not data_to_process:
            logger.info("All items have already been evaluated.")
        else:
            logger.info(f"Total items to evaluate: {len(data_to_process)}")
            evaluator.evaluate(data_to_process, save_file_path, save_file_brief_path)

        if args.perform_analysis:
            analyze_results(save_file_path, args.analysis_filter)

if __name__ == "__main__":
    main()
