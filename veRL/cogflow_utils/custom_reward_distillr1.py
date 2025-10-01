from transformers import AutoTokenizer

# BEGIN: main parameters
# Param1: omega_1(response reward), omega_2(length reward), omega_3(diversity reward)
omega = [1, 0.1, 0.]
 
# Param2: do not change
scale_factor = [4, 4]

# Param3: path to your tokenizer
TOKENIZER_MODEL = "models/llama_sft_distillr1" 
# END: main parameters

print("Loading tokenizer inside my_rewards.py module...")
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
print(f"Tokenizer '{TOKENIZER_MODEL}' loaded globally in custom reward module.")

def check_think_format(output: str) -> bool:
    all_possible_label_content = ['think', 'Observation', 'Attribution', 'Motivation', 'Regulation', 'Efficacy', 'Behavior', 'Identification']
    all_valid_label_content = ['Observation', 'Attribution', 'Motivation', 'Regulation', 'Efficacy', 'Behavior']
    
    split_bthink = output.split('<think>')
    if len(split_bthink) != 2:
        print(f"INVALID FORMAT: <think> occur time incorrect. ")
        return False
    if split_bthink[0].replace("\n", "").replace(" ", "") != "":
        print(f"INVALID FORMAT: content before <think> is not empty. ")
        return False
    output = split_bthink[1]
    
    split_ethink = output.split('</think>')
    if len(split_ethink) != 2:
        print(f"INVALID FORMAT: </think> occur time incorrect: ")
        return False
    output = split_ethink[0]
    response = split_ethink[1]
    for label_content in all_possible_label_content:
        if f"<{label_content}>" in response:
            print(f"INVALID FORMAT: <{label_content}> found in response. ")
            return False
        if f"</{label_content}>" in response:
            print(f"INVALID FORMAT: </{label_content}> found in response. ")
            return False
        
    return True

def get_length(response_str: str) -> tuple[int, int]:
    split = response_str.split('</think>')
    return len(TOKENIZER.encode(split[0])), len(TOKENIZER.encode(split[1]))

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