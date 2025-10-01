from evaluate_template import process_direct_score_template
from chain_eval_utils import convert_chain_to_str
from chain_eval_utils import call_api_json_repeat

format_dict = {
    "think": "evaluation process",
    "evaluation_result": {
        "coherence": {
            "reason": "Explain your reasoning", 
            "score": 1
        },
        "interpretability": {
            "reason": "Explain your reasoning", 
            "score": 1
        },
        "predictability": {
            "reason": "Explain your reasoning", 
            "score": 1
        }
    }
}

def score_by_process(user_input, all_chains: list, args, top_k = None) -> list:
    for chain in all_chains:
        chain[-1]["additional"]["reasoning_process"] = convert_chain_to_str(chain[:-1])
        prompt = process_direct_score_template.format(
            reasoning_flow = chain[-1]["additional"]["reasoning_process"],
        )
        result = call_api_json_repeat(
            prompt=prompt, 
            args=args, 
            format_dict=format_dict,
            format_level=-1,
            brace_type='{',
            keep_reason=False
        )
        scores = [result["evaluation_result"][key]["score"] for key in result["evaluation_result"]]
        chain[-1]["additional"]["process_score"] = min(scores)
        chain[-1]["additional"]["process_judge"] = result
    return all_chains