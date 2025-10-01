import re
import json
import random
import os
from datetime import datetime
import itertools
import time
import copy
import numpy as np


import filelock
input_tokens = 0
output_tokens = 0
fail_wait_secs = 70
retry_time = 10
token_file_name = "token_cnt.txt"

lock_file_path = "./tmp/token_cnt_lock"
token_cnt_lock = filelock.FileLock(lock_file_path, timeout=20)

from api_config import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def check_time_in_discount_period(current_time_hour, current_time_minute, buffer_minutes = 0):
    # deepseek dicount from utc 16:30 to 0:30
    if current_time_hour > 16 and current_time_hour < 24:
        return True
    elif current_time_hour == 0 and current_time_minute < 30-buffer_minutes:
        return True
    elif current_time_hour == 16 and current_time_minute > 30:
        return True
    else:
        return False

def get_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_cartesian_dict(dict_of_lists):
    keys = dict_of_lists.keys()
    values = dict_of_lists.values()
    cartesian_product = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return cartesian_product

def extract_last_quoted_text(text):
    matches = re.findall(r'"(.*?)"', text)
    return matches[-1] if matches else None

def extract_outermost_braced_text(text, brace1 = '{', brace2 = '}'):
    cnt = 0
    result = ''
    for char in text:
        if char == brace1:
            cnt += 1
        if cnt > 0: result += char
        if char == brace2:
            cnt -= 1
            if cnt == 0: break
    if result == '': return text
    return result

def prepare_for_json_loads(text, brace_type = '{'): 
    brace1 = '{'
    brace2 = '}'
    if brace_type == '[': 
        brace1 = '['
        brace2 = ']'
    text = extract_outermost_braced_text(text, brace1, brace2)
    # text = text.replace("“", "\"")
    # text = text.replace("”", "\"")
    while text[0] != '\"' and text[0] != brace1: text = text[1:]
    while text[-1] != '\"' and text[-1] != brace2: text = text[:-1]
    if text[0] != brace1: text = brace1 + text
    if text[-1] != brace2: text = text + brace2
    return text

def prepare_for_yaml_load(text):
    while text[:7] != '```yaml':
        text = text[1:]
    while text[-3:] != '```':
        text = text[:-1]
    text = text[7:-3]
    return text


def token_cnt(url, model_name, input_token, output_token, time_span):
    url = "default_2"
    with token_cnt_lock:
        if model_name in ["deepseek-reasoner", "ds-r1"] and check_time_in_discount_period(datetime.now().hour, datetime.now().minute):
            model_name += "_discount"
        try:
            with open(token_file_name, "r") as f:
                data = json.load(f)
        except:
            print(f"[[no proper {token_file_name}, recreate]]")
            data = {}
        if not url in data:
            data[url] = {}
        if not model_name in data[url]:
            data[url][model_name] = {
                "input": 0, 
                "output": 0, 
                "time": 0, 
                "cnt": 0
            }
        data[url][model_name]["input"] += input_token
        data[url][model_name]["output"] += output_token
        data[url][model_name]["time"] += time_span
        data[url][model_name]["cnt"] += 1
        with open(token_file_name, "w") as f:
            json.dump(data, f)

def call_api_general(prompt: str, args):

    platform = args.platform if not args.platform is None else "custom"
    if platform == "bingxing":
        model_name_map = model_name_map_bingxing
        client = bingxing_client
    elif platform == "deepseek":
        model_name_map = model_name_map_deepseek
        client = deepseek_client
    elif platform == "silicon_flow":
        model_name_map = model_name_map_silicon
        client = siliconflow_client
    elif platform == "glm":
        model_name_map = model_name_map_glm
        client = zhipuai_client
    elif platform == "custom":
        model_name_map = model_name_map_custom
        client = custom_client
    else:
        raise Exception("platform not supported")

    print(f"client api = {client.api_key}")
    
    arg_model = args.model if not args.model is None else "default"
    model = model_name_map.get(arg_model, model_name_map["default"])
    
    if args.check_generate_time == "yes":
        hour = datetime.now().hour
        minute = datetime.now().minute
        if not check_time_in_discount_period(hour, minute, 20):
            print("Out of time region, force exit. please check run_all_report.txt for current status. ")
            exit()
    
    try:
        base_url = client.base_url
    except:
        base_url = client._base_url
    print(f"call {model} at {base_url}")
    time_1 = time.perf_counter()
    if hasattr(args, 'temperature'):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ], 
            temperature=args.temperature, 
        )
    else: 
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ], 
        )
    time_2 = time.perf_counter()
    with open("response.txt", "w") as f:
        f.write(f"{response}")
    
    # BEGIN COUNTER
    global input_tokens, output_tokens
    input_token = response.usage.prompt_tokens
    output_token = response.usage.completion_tokens
    input_tokens += input_token
    output_tokens += output_token
    print(f"token used: input: {input_tokens}, output: {output_tokens}")
    token_cnt(base_url, model, input_token, output_token, time_2-time_1)
    # END COUNTER

    if hasattr(response.choices[0].message, 'reasoning_content'):
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return {
            "reasoning_content": reasoning_content, 
            "content": content
        }
    else:
        content = response.choices[0].message.content
        return content

def call_api_manual(prompt, args):
    with open("manual.txt", "w", encoding="utf-8") as f:
        print(prompt, file=f)
    res = input("prompt is in manual.txt, please enter the response: ")
    return res

def call_api(prompt, args, brace_type = '{', prepare_for_json = True):
    with open("prompt.txt", "w", encoding="utf-8") as f:
        print(prompt, file=f)
    time.sleep(0.1)

    if args.model == "manual":
        api_func = call_api_manual
    else:
        api_func = call_api_general
    res = api_func(prompt, args)        
    if prepare_for_json: 
        if isinstance(res, dict):
            res["content"] = prepare_for_json_loads(res["content"], brace_type = brace_type)
        else:
            res = prepare_for_json_loads(res, brace_type = brace_type)
    return res

def call_api_repeat(prompt, args, brace_type = '{',prepare_for_json = True):
    """
    Calls an API with an automatic retry mechanism.

    Args:
        prompt (str): The input prompt string. Only user prompts are supported, not system prompts.
        args (argparse.Namespace): An object containing the necessary arguments. It requires
            the following three attributes:
                model (str): The name of the model (e.g., "qwen3_32B", "gpt", "ds-r1", "ds-v3").
                platform (str): The name of the platform (e.g., "silicon_flow", "deepseek", "glm").
                                This can also be a custom value.
                check_generate_time (str): Whether to check the generation time. Accepts "yes" or "no".
        prepare_for_json (bool, optional): Whether to prepare the returned string for JSON conversion.
                                           Defaults to True.
        brace_type (str, optional): If `prepare_for_json` is True, this specifies the type of brace
                                    to look for when extracting the JSON object. Defaults to '{'.

    Returns:
        dict or str: If a reasoning model is used, returns a dictionary containing "reasoning_content"
                     and "content" keys, each with a string value. Otherwise, returns the raw
                     response string directly.
    """
    for i in range(retry_time):
        try: 
            res = call_api(prompt, args, brace_type=brace_type, prepare_for_json=prepare_for_json)
            break
        except Exception as e:
            prompt += "\n."
            print(f"call_api_repeat(No. {i}): {e}")
            time.sleep(fail_wait_secs)
    return res

def check_nested_keys(template: dict, target: dict) -> bool:
    """
    recursively check if all the keys in template are in target 
    """
    if not template.keys() <= target.keys():
        return False

    for key in template:
        if isinstance(template[key], dict):
            if isinstance(target[key], dict):
                if not check_nested_keys(template[key], target[key]):
                    return False
            else:
                return False

    return True

def call_api_json_repeat(prompt, args, format_dict = None, info = "", format_level = 2, brace_type = '{', keep_reason=False):
    for i in range(retry_time):
        try: 
            res = call_api(prompt, args, brace_type=brace_type)
            if isinstance(res, dict):
                if keep_reason: 
                    res["content"] = json.loads(res["content"])
                else:
                    res = json.loads(res["content"])
            else:
                res = json.loads(res)
            if format_dict is not None:
                if format_level == 2: 
                    for name in format_dict:
                        assert name in res
                        if isinstance(format_dict[name], dict):
                            for item in res[name]: 
                                assert item in format_dict[name]
                            for item in format_dict[name]: 
                                assert item in res[name]
                elif format_level == -1:
                    assert check_nested_keys(format_dict, res)
                else: # level == 1
                    for name in res:
                        assert name in format_dict
                    for name in format_dict:
                        assert name in res
            break
        except Exception as e:
            prompt += "\n."
            print(f"call_api_json_repeat(No. {i}): {info}, {e}")
            time.sleep(fail_wait_secs)
    return res
