import argparse

from chain_eval_utils import *
from chain_output_eval import score_by_output
from chain_process_eval import score_by_process

def score_by_length(user_input, all_chains: list, args, top_k = None) -> list:
    # calculate the length
    all_length = []
    for chain in all_chains:
        reasoning = convert_chain_to_str(chain[:-1])
        chain[-1]["additional"]["reason_length"] = len(reasoning)
        all_length.append(chain[-1]["additional"]["reason_length"])

    # calculate normalized length as the score
    # shorter, better
    len_min = np.min(all_length)
    len_max = np.max(all_length)
    for chain in all_chains:
        chain[-1]["additional"]["length_score"] = (len_max-chain[-1]["additional"]["reason_length"]) / (len_max - len_min+1)
    return all_chains

def cal_score_and_select_best_1(user_input, all_chains: list, args, top_k = None, w1 = 0.6, w2 = 0.3, w3 = 0.1):
    # param: w3 = 1-w1-w2
    # input: [[{node1}, {node2}, ...], [..], ..]
    # output: [{node1}, {node2}, ...]
    
    # get 'output_score' in 'additional' part of terminate
    all_chains, all_output_explain = score_by_output(user_input, all_chains, args, top_k = top_k, type="mid")
    # get 'process_score' in 'additional' part of terminate
    all_chains = score_by_process(user_input, all_chains, args)
    # get 'length_score' in 'additional' part of terminate
    all_chains = score_by_length(user_input, all_chains, args)
    
    output_tmp(json.dumps(all_chains, indent=4, ensure_ascii=False), "all scores calculated")

    # get 'score' in 'additional' part of terminate
    for chain in all_chains:
        chain[-1]['additional']['score'] = w1*chain[-1]['additional'].get('output_score', 0) + w2*chain[-1]['additional'].get('process_score',0) + w3*chain[-1]['additional'].get('length_score',0)
    
    output_tmp(json.dumps(all_chains, indent=4, ensure_ascii=False), "tot scores calculated")
    
    return (all_chains, all_output_explain)

def select_best_and_store(datum, args):
    user_input, all_chains, constraint = extract_user_input_chains(datum, args)
    user_input = user_input+'\n'+constraint
            
    all_chains, all_output_explain = cal_score_and_select_best_1(user_input, all_chains, args)
    
    datum["output_explain"] = all_output_explain
    
    for chain in all_chains:
        for node in datum["reason_nodes"]:
            if node["id"] == chain[-1]["id"]:
                node["additional"] = chain[-1]["additional"]
    
    return datum

def main(args):
    random.seed(args.seed)
    
    with open(os.path.join(args.folder_path, args.file_name+".json"), "r") as f:
        data = json.load(f)
    
    result_folder = os.path.join(args.folder_path, args.file_name+"_eval")
    os.makedirs(result_folder, exist_ok=True)
    for idx, datum in enumerate(data):
        cur_datum = select_best_and_store(datum, args)
        file_name = get_time_str() + f"_num{idx}.json"
        print(f"--- No. {idx} over ---")
        with open(os.path.join(result_folder, file_name), "w") as f:
            json.dump(cur_datum, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # generate control
    parser.add_argument("--model", type=str, default="ds-r1")
    parser.add_argument("--platform", type=str, default="custom")
    parser.add_argument("--check_generate_time", type=str, default="no", choices=["yes", "no"])
    # evaluater control
    parser.add_argument("--baseline_model", type=str, default="ds-r1")
    # dataset control
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)