from utils_4 import *
from prompt_template_json import *
# from response_constraints import ResponseConstraint

import copy
import json
import os
import argparse

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def get_version(args):
    return f"CogFlow_{args.model}_{args.version}"

def load_dataset(args):
    print("in load_dataset")
    # load dataset from file
    if args.dataset_type == "json":
        with open(os.path.join(args.dataset_folder, args.dataset_name+".json"), "r") as f:
            data = json.load(f)
    elif args.dataset_type == "jsonl": 
        data = load_jsonl(os.path.join(args.dataset_folder, args.dataset_name+".jsonl"))
    else: 
        assert False, "not completed"
        
    # process the dataset
    result = data[args.data_from : args.data_to]
    return result

def get_chain_from_reason_nodes(nodes: list, tar_num) -> list:
    # input tree
    # output chain
    result = []
    while tar_num >= 0:
        for node in nodes:
            if node["id"] == tar_num:
                result.append(node)
                tar_num = node["parent"]
                break
    result.reverse()
    return result

def convert_chain_to_str(nodes: list) -> str:
    # input:  [{"id": 0, "parent": -1, "type": "6 types / basic / result", "content": "..."}]
    # output: node 1 (name):\n ... \n ...
    result = ""
    for i in range(len(nodes)):
        cur_node_str = f"[{nodes[i]['type']}]:{nodes[i]['content']}\n"
        result += cur_node_str
    return result

def get_cog_reaon_NBN(data: list, args) -> dict:
    # from [{"scenario": "...", ...}]
    # to [{"scenario": "...", ..., "reason_nodes": [{"id": 0, "parent": -1, "type": "6 types / Basic / Result", "content": "...", "additional": {}}]]
    all_possible_nodes = list(cog_ordinary_NBN_content_node_en.keys())
    output_timestamp = args.dataset_name+datetime.now().strftime("%Y%m%d_%H%M%S")
    for datum in data:
        # get scenario and question
        scenario = datum["extracted"]["scenario"]
        question = datum["extracted"]["question"]
            
        # get user input
        user_input = scenario+"\n"+question
        constraint_prompt = datum['extracted'].get('constraint', '')
        
        # init reason_nodes
        reason_nodes = []
        unexplored_nodes = {0}
        reason_nodes.append({
            "id": 0, 
            "parent": -1, 
            "type": "Observation", 
            "content": "none", 
            "additional": {}
        })
        
        # start bfs
        # at most 50 nodes
        def extend_node(cur_node_id):
            unexplored_nodes.remove(cur_node_id)
            # generate content of reason_nodes[cur_node_id]
            # result format: {"content": "..."}
            node_name = reason_nodes[cur_node_id]["type"]
            fa_node_id = reason_nodes[cur_node_id]["parent"]
            print(node_name)
            
            # get content of prompt
            node_description = cog_ordinary_NBN_content_node_en[node_name]
            previous_nodes = get_chain_from_reason_nodes(reason_nodes, fa_node_id)
            previous_nodes = convert_chain_to_str(previous_nodes)
            analyze_expect = ""
            if fa_node_id >= 0:
                analyze_expect = reason_nodes[fa_node_id]["additional"]["children_reason"]
            
            # prepare the prompt
            if node_name == "Terminate": 
                curr_user_input = user_input+"\n"+constraint_prompt
                prompt = cog_ordinary_NBN_result_full_template_en.format(
                    user_input = curr_user_input, 
                    previous_nodes = previous_nodes, 
                    node_name = node_name, 
                    node_description = node_description, 
                    analyze_expect = analyze_expect
                )
            else: 
                curr_user_input = user_input+"\n"+constraint_prompt
                prompt = cog_ordinary_NBN_content_template_en.format(
                    user_input = curr_user_input, 
                    previous_nodes = previous_nodes, 
                    node_name = node_name, 
                    node_description = node_description, 
                    analyze_expect = analyze_expect
                )
            
            # call api and format the result
            for _ in range(5):
                try:
                    if node_name != "Terminate":
                        result = call_api_json_repeat(prompt, args, keep_reason=True)
                        if isinstance(result, dict): 
                            reason_nodes[cur_node_id]["additional"]["r1-reason"] = result["reasoning_content"]
                            result = result["content"]
                    else:
                        result = call_api_repeat(prompt, args, prepare_for_json=False)
                        if isinstance(result, dict):
                            reason_nodes[cur_node_id]["additional"]["r1-reason"] = result["reasoning_content"]
                            result = {
                                "content": result["content"]
                            }
                        else:
                            result = {
                                "content": result
                            }

                    reason_nodes[cur_node_id]["content"] = result["content"]
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)
                reason_nodes[cur_node_id]["content"] = ""
            
            # if node 'Terminate', end
            if reason_nodes[cur_node_id]["type"] != "Terminate":
                # get children nodes
                # result format: {"rationale": "...", "next_nodes": ["...", ...]}
                previous_nodes = get_chain_from_reason_nodes(reason_nodes, cur_node_id)
                previous_nodes = convert_chain_to_str(previous_nodes)
                curr_user_input = user_input+"\n"+constraint_prompt
                prompt = cog_ordinary_NBN_choose_template_en.format(
                    user_input = curr_user_input, 
                    previous_nodes = previous_nodes
                )
                result = call_api_json_repeat(prompt, args)
                
                reason_nodes[cur_node_id]["additional"]["children_reason"] = result["rationale"]
                reason_nodes[cur_node_id]["additional"]["children"] = result["next_step_candidates"]
                children_id = []
                
                for n_node in result["next_step_candidates"]:
                    num = len(reason_nodes)
                    if not n_node in all_possible_nodes:
                        continue
                    reason_nodes.append({
                        "id": num, 
                        "parent": cur_node_id, 
                        "type": n_node, 
                        "content": "none", 
                        "additional": {}
                    })
                    children_id.append(num)
                    unexplored_nodes.add(num)
                    
                reason_nodes[cur_node_id]["additional"]["children_id"] = copy.deepcopy(children_id)
            
            # output the half-completed tree now
            datum["reason_nodes"] = reason_nodes
            os.makedirs(os.path.join(args.result_folder, get_version(args)), exist_ok=True)
            with open(os.path.join(args.result_folder, get_version(args), output_timestamp+".json"), "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        
        for chain_cnt in range(0,8):
            real_node_cnt = len(reason_nodes) - len(unexplored_nodes)
            if real_node_cnt >= 35:
                break
            if len(unexplored_nodes) == 0:
                break
            print(f"Generating Chain No.{chain_cnt}")
            cur_root = random.choice(list(unexplored_nodes))
            for _ in range(20):
                extend_node(cur_root)
                if reason_nodes[cur_root]["type"] == "Terminate":
                    break
                cur_root = reason_nodes[cur_root]["additional"]["children_id"][0]
        
        all_models = [args.model]
        all_potentials = args.reference_models
        for model in all_potentials:
            if not model in all_models:
                all_models.append(model)
        for model in all_models:
            args.model = model
            prompt = user_input+"\n"+constraint_prompt
            try:
                result = call_api_repeat(prompt, args, prepare_for_json=False)
                if isinstance(result, dict):
                    reason_result = result["reasoning_content"]
                    final_result = result["content"]
                else:
                    reason_result = ""
                    final_result = result
            except:
                reason_result = "error"
                final_result = "error"
            args.model = all_models[0]
            reason_nodes.append({
                "id": len(reason_nodes), 
                "parent": -1, 
                "type": model, 
                "content": reason_result, 
                "additional": {
                    "children_id": [len(reason_nodes)+1]
                }
            })
            reason_nodes.append({
                "id": len(reason_nodes), 
                "parent": len(reason_nodes)-1, 
                "type": "Terminate", 
                "content": final_result, 
                "additional": {}
            })
            # output the half-completed tree now
            datum["reason_nodes"] = reason_nodes
            os.makedirs(os.path.join(args.result_folder, get_version(args)), exist_ok=True)
            with open(os.path.join(args.result_folder, get_version(args), output_timestamp+".json"), "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        args.model = all_models[0]
    
    with open(args.app_output_info, "w") as f:
        f.write(output_timestamp)
    return data

def main():
    parser = argparse.ArgumentParser()
    # generate control
    parser.add_argument("--model", type=str, default="ds-r1")
    parser.add_argument("--platform", type=str, default="custom")
    parser.add_argument("--reference_models", type=str, default='ds-r1,ds-v3')
    # dataset control
    parser.add_argument("--dataset_type", type=str, default="jsonl", choices=["json", "jsonl"])
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_folder", type=str, default="dataset")
    parser.add_argument("--result_folder", type=str, default="result")
    parser.add_argument("--data_from", type=int, default=0)
    parser.add_argument("--data_to", type=int, default=3)
    parser.add_argument("--app_output_info", type=str, default="app_output_info.tmp")
    parser.add_argument("--check_generate_time", type=str, default="no", choices=["yes", "no"])
    
    parser.add_argument("--version", type=str, default="6")
    
    args = parser.parse_args()
    # change reference_models to list
    args.reference_models = args.reference_models.split(",")
    data = load_dataset(args)
    get_cog_reaon_NBN(data, args)
    
if __name__ == '__main__':
    main()