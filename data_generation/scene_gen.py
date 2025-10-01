import re
import json
import random
import os
from datetime import datetime
import time
import itertools
import argparse
from typing import Tuple

from utils_4 import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_reference_scene():
    file = "dataset/scene_examples.jsonl"
    with open(file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    random.shuffle(data)
    results = []
    for item in data[:3]:
        example = {
            "scenario": item['STORY'], 
            "question": item['QUESTION'],
        }
        results.append(example)
    return results

generate_prompt_template = \
"""## **[Task]**

Given a scenario description and suggestions related to the scenario, you are required to generate a scenario and a question for the COGNITION TEST. You should just use the description and suggestions as triggers; you can convert the scenario arbitrarily by yourself:

1. **Summarize the Scenario**:  
    - Objective: Craft a scene of dynamic social interaction focusing on several with motion-driven engagement. Describe the scenario using plain words. 
    - It should focus on social interactions, with enough details, for example: 
        - Environmental Context: Describe a specific time/place.
        - Specific Task: Clearly state efficient information. For example, the problem they are facing or the activity they are doing. 
        - Character Relationship: Clearly state the relationship between the roles. 
        - Character Dynamics: Establish clear profiles of the characters. 
    - IMPORTANT: The scenario should be concise with enough details (not necessarily related to the original scenario or the question). It's better to include either relevant or irrelevant details to the question stated in the next step. 

2. **State the question**:
    - Concisely state the question based on the scenario in one sentence using the third person perspective. But you should only state the question simply, using words of mouth. 
    - You should double-check that the answer is NOT stated in the scenario. There should also be no direct/indirect hints in the scenario. 
    - The question should be suitable for a cognitive test. You should ignore the original question stated in [Scenario Description]. 

3. **Output Format**:  
    Present your result in JSON format using the following structure and respond in English. You should only use simpler vocabulary at a high school level to form your answers. Make sure that quotes inside all strings are escaped with backslashes:  
```json
{{
    "scenario": "Scenario Summary", 
    "question": "Question in one sentence"
}}
```  

## **#Possible Tests#**  
These examples are way too brief and easy; your output should be more detailed and harder. For example, you should not give any hints, and it had better be open-ended. 
```json
{examples}
```

## **[Scenario Description]**  
{scenario_description}  

## **[Suggestion]**  
{suggestion}"""

get_constraint_template = \
"""## **[Task]**

Given a [User Input] containing a Story and an open-ended Question, please propose an appropriate constraint on the output format for the answer to the Question containing all constraints in [Required Constraints]. The proposed constraint should be concise.

Note:
- The generated instructions cannot contain any content related to or hinting at the answer. 
- The output format should follow [Output Format].

## **[User Input]**
{user_input}

## **[Required Constraints]**
{required_constraints}

## **[Output Format]**
You should directly output the instruction as natural sentences without any additional words or explanations (especially explain how you generate the output). In one word, your whole output can be directly used as a constraint.
"""

check_validity_template = \
"""## **[Task]**

Please check the scenario, question, and constraint given in the [User Input], determine in order whether the following conditions are met, and provide your response following the output requirements in [Check Output Format].

1. Please check if the question is relevant to the scenario. For example, the content involved must be mentioned in the scenario.
2. Please check if the constraint does not imply the answer to the question, but only provides formatting content or restates the content of the question.
3. Please check if the constraint does not contain confusing content. The constraint must be a reasonable format restriction for someone answering the question. For example, the act of "requiring in the constraint not to imply the answer" does not meet the requirement.

## **[User Input]**

{user_input}

## **[Check Output Format]**

Please use JSON format for the output, with only one key named 'result', and the value being a boolean type. true indicates that all requirements are met, and false indicates that at least one requirement is not met. Please follow the structure below:

```json
{{
	"result": true / false
}}
```"""

def check_validity(args, extracted: dict[str, str]) -> bool:
    prompt = check_validity_template.format(
        user_input = json.dumps(extracted, ensure_ascii=False, indent=2),
    )
    validity = call_api_json_repeat(prompt, args, format_dict={"result": True}, format_level=1, brace_type='{')
    return validity["result"]

def get_constraint(args, user_input: str) -> Tuple[str, list]:
    format_constraints = [
        "The output should be formated in JSON / YAML / Markdown/ Bullet or any other suitable format. To state this constraint, you should choose one specific format, and give a brief demonstration of the format of the output to make it clear. Note that your instructions should be concise. **IMPORTANT: You should make sure that your demonstration is just formal, without any hint of the real answer. **",
    ]
    verbosity_constraints = [
        "The output should be of high verbosity, which means detailed (but still need to be concise). ", 
        "The output should be of medium verbosity, which means balancing between brief and detailed. ",
        "The output should be of low verbosity, which means brief and concise. ",
    ]
    all_constraints = [format_constraints, verbosity_constraints]
    active_constraints = []
    for constraint in all_constraints:
        if random.randint(1, 100) <= 70:
            active_constraints.append(constraint)
    random.shuffle(active_constraints)
    if len(active_constraints) == 0:
        return "", active_constraints
    required_constraints_list = [random.choice(item) for item in active_constraints]
    required_constraints = "".join([f"{i+1}: {item}\n" for i, item in enumerate(required_constraints_list)])
    prompt = get_constraint_template.format(
        user_input = user_input,
        required_constraints = required_constraints,
    )
    old_model = args.model
    args.model = args.model_constraint
    constraint = call_api_repeat(prompt, args, prepare_for_json=False)
    args.model = old_model
    if isinstance(constraint, dict):
        constraint = constraint['content']
    return constraint, required_constraints_list

def comb_ori_ext_eval_summary(args, num, situation, previous = None):
    refined = "no"
    descrption = situation["Post Text"]
    suggestion = situation["Comments"]
    examples = get_reference_scene()
    
    prompt = generate_prompt_template.format(
        scenario_description = descrption,
        suggestion = suggestion,
        examples = json.dumps(examples, indent=4, ensure_ascii=False)
    )
    extracted = call_api_json_repeat(prompt, args)
    user_input = extracted["scenario"]+"\n"+extracted["question"]
    constraint, constraint_original = get_constraint(args, user_input)

    extracted['constraint'] = constraint

    extract_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{num}_{refined}".replace(" ", "_").replace("(", "").replace(")", "").replace("'", "_")
    return {
        "original": situation,
        "extracted": extracted,
        "summary": {
            "refined": refined,
            "extract_time": extract_time, 
            "name": name, 
            "constraint_original": constraint_original,
            "examples": examples,
        }
    }
    
def comb_ori_ext_eval_summary_retry(args, num, situation, previous = None):
    for i in range(3):
        result = comb_ori_ext_eval_summary(args, num, situation, previous)
        if check_validity(args, result['extracted']):
            result['summary']['refined_time'] = i
            return result
        logger.info(f"result invalid, retrying...")
        file_name = "tmp/invalid.json"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                invalid = json.load(f)
        else:
            invalid = []
        invalid.append(result)
        with open("tmp/invalid.json", "w") as f:
            json.dump(invalid, f, ensure_ascii=False, indent=2)
        time.sleep(2)

def get_score_list(unit):
    res = []
    for item in unit["evaluation"].values():
        res.append(item["score"])
    return res

def situation_shuffle(situations):
    '''average select situations based on Subreddit'''
    if len(situations) == 0:
        return situations
    if "Subreddit" not in situations[0]:
        return situations
    
    list_situation_by_subreddit = {}
    for situation in situations:
        subreddit = situation["Subreddit"]
        if subreddit not in list_situation_by_subreddit:
            list_situation_by_subreddit[subreddit] = []
        list_situation_by_subreddit[subreddit].append(situation)
    for subreddit in list_situation_by_subreddit:
        random.shuffle(list_situation_by_subreddit[subreddit])
    res = []
    longest = max([len(list_situation_by_subreddit[subreddit]) for subreddit in list_situation_by_subreddit])
    for i in range(longest):
        for subreddit in list_situation_by_subreddit:
            if i < len(list_situation_by_subreddit[subreddit]):
                res.append(list_situation_by_subreddit[subreddit][i])
    return res

def main(args):
    assert args.dataset_src == "reddit"
    
    dataset_folder = "dataset"
    dataset_name = "reddit_raw"
    output_name = f"{dataset_name}_{args.start_num}-{args.end_num}"
    logger.info(f"save to {output_name}")
    
    situation_file = f"./{dataset_folder}/{dataset_name}.json"
    origin_result_file = f"./{dataset_folder}/output/{output_name}.json"
    os.makedirs(os.path.dirname(origin_result_file), exist_ok=True)
    
    with open(situation_file, "r", encoding='utf-8') as f:
        situations = json.load(f)
    origin_results = []
    random.seed(34+args.start_num)
    random.shuffle(situations)
    situations = situation_shuffle(situations)
    
    for id, situation in enumerate(situations[args.start_num:args.end_num]):
        origin_result = comb_ori_ext_eval_summary_retry(args, id, situation, None)
        origin_results.append(origin_result)
        with open(origin_result_file, "w", encoding='utf-8') as f:
            json.dump(origin_results, f, ensure_ascii=False, indent=4)
        
    with open(args.app_output_info, "w") as f:
        f.write(output_name + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="ds-r1")
    parser.add_argument("--model_constraint", type=str, default="ds-v3")
    parser.add_argument("--dataset_src", type=str, default="reddit", choices=['reddit'])
    parser.add_argument("--start_num", type=int, default=0)
    parser.add_argument("--end_num", type=int, default=1)
    
    parser.add_argument("--app_output_info", type=str, default="app_output_info.txt")
    parser.add_argument("--check_generate_time", type=str, default="no", choices=["yes", "no"])
    parser.add_argument("--platform", type=str, default="custom")

    args = parser.parse_args()
    main(args)
