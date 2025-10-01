from transformers import AutoTokenizer

# BEGIN: main parameters
# Param1: omega_1(response reward), omega_2(length reward), omega_3(diversity reward)
omega = [1, 0., 0.]
 
# Param2: do not change
scale_factor = [4, 4]

# Param3: path to your tokenizer
TOKENIZER_MODEL = "models/llama_sft_noreason" 
# END: main parameters

print("Loading tokenizer inside my_rewards.py module...")
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
print(f"Tokenizer '{TOKENIZER_MODEL}' loaded globally in custom reward module.")

def check_think_format(output: str) -> bool:
    return True

def get_length(response_str: str) -> tuple[int, int]:
    split = response_str.split('</think>')
    return len(TOKENIZER.encode("")), len(TOKENIZER.encode(response_str))

from custom_reward_utils import compute_score_2

def compute_score(
    solution_str, 
    ground_truth, 
    data_source, 
    extra_info, 
    rm_scores: float, 
    correctness: float, 
    entropy_info: dict, 
):
    result = compute_score_2(
        solution_str, ground_truth, data_source, extra_info, rm_scores, correctness, entropy_info, 
        check_think_format, 
        get_length,
        omega, 
        scale_factor,
    )
    
    # you can output debugging info here...

    return result