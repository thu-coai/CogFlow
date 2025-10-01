import math
import logging
from collections.abc import Callable

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

def sigmoid(x):
    if x < 0:
        return math.exp(x) / (1+math.exp(x))
    else:
        return 1 / (1 + math.exp(-x))

# see: verl/workers/reward_manager/naive2.py
def compute_score_2(
    solution_str, ground_truth, data_source, extra_info, rm_scores: float, correctness: float, entropy_info: dict, 
    check_think_format: Callable[[str], bool], 
    get_length: Callable[[str], tuple[int, int]],
    omega: list[float], 
    scale_factor: list[float], 
):
    format_valid = check_think_format(solution_str)
    if not format_valid:
        return {
            'score': 0,
            'format_check': 0, 
            'correctness': correctness,
            
            'rm_score': rm_scores,
            'rm_score_valid': -998244353,
             
            'len_score': -998244353,
            'len_reason': -998244353,
            'len_response': -998244353,
            'len_reason_reflong': -998244353,
            'len_reason_refshort': -998244353,
            
            'diversity_score': entropy_info['node_entropy_score'],
            'entropy': entropy_info['node_entropy'], 
        }
        
    sol_len = get_length(solution_str)

    if data_source not in ['cog_flow']:
        length_max = 10000
        length_buffer = 2000
        length_real = sol_len[0] + sol_len[1]
        length_score = 1 if length_real <= length_max - length_buffer else ((length_max - length_real) / length_buffer if length_real < length_max else 0)
        
        score = 0.8*correctness + 0.1*length_score + 0.1*entropy_info['node_entropy_score']
        # print(f"in RLVR task, {data_source}, correctness = {correctness}, score = {score}, length = {length_real}")
        return {
            'score': length_score,
            'format_check': 1, 
            'correctness': correctness,
            
            'rm_score': rm_scores, 
            'rm_score_valid': rm_scores,
            
            'len_score': length_score,
            'len_reason': sol_len[0],
            'len_response': sol_len[1],
            'len_reason_reflong': -998244353,
            'len_reason_refshort': -998244353,

            'diversity_score': entropy_info['node_entropy_score'],
            'entropy': entropy_info['node_entropy'],
        }
    
    raw_answer = rm_scores
    sol_len = get_length(solution_str)
    ref_long = extra_info['reference']['len_reason_long']
    ref_short = extra_info['reference']['len_reason_short']
    left_mid = ref_short/2
    left_len = ref_short/2
    right_mid = ref_long+ref_short
    right_len = ref_short/2

    if left_len <= 1:
        logger.warning(f"left_len <= 1: {left_len}")
        left_len = 2
    if right_len <= 1:
        logger.warning(f"right_len <= 1: {right_len}")
        right_len = 2
    len_score = sigmoid(scale_factor[0]*(sol_len[0] - left_mid)/(left_len+1e-6)) * sigmoid(scale_factor[1]*(right_mid - sol_len[0])/(right_len+1e-6))
    
    diversity_score = entropy_info['node_entropy_score']
    
    answer = omega[0] * raw_answer # Res reward
    answer += omega[1] * len_score # Len reward
    answer += omega[2] * diversity_score # Div reward
    
    return {
        'score': answer,
        'format_check': 1, 
        'correctness': correctness,
        
        'rm_score': rm_scores, 
        'rm_score_valid': rm_scores,
        
        'len_score': len_score,
        'len_reason': sol_len[0],
        'len_response': sol_len[1],
        'len_reason_reflong': ref_long,
        'len_reason_refshort': ref_short,
        
        'diversity_score': entropy_info['node_entropy_score'],
        'entropy': entropy_info['node_entropy'],
    }
