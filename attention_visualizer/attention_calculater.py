# -*- coding: utf-8 -*-
"""
This script performs causal attention analysis and saves the results to a file
for offline visualization. This version focuses solely on <tag>...</tag> nodes.

Key features:
1. get_full_attention: Calculates the full attention matrix for all tokens.
2. find_node_boundaries: Identifies token spans for nodes using a simplified regex.
3. segment_into_sentences: Segments text into sentence-like units.
4. save_attention_data: Saves all relevant data into a single .npz file.
"""
import os
import re
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def find_node_boundaries(full_text: str, prompt_text: str, tokenizer) -> list[dict]:
    """
    Finds token indices for nodes using the <tag>content</tag> format.
    """
    response_text = full_text[len(prompt_text):]
    encoding = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding.pop("offset_mapping").squeeze().tolist()

    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    search_text, char_offset = (think_match.group(1), len(prompt_text) + think_match.start(1)) if think_match else (response_text, len(prompt_text))

    # The pattern now only looks for the <tag>content</tag> structure.
    tag_pattern = r"<([a-zA-Z_ -]+)>(.*?)</\1>"

    boundaries = []
    for match in re.finditer(tag_pattern, search_text, re.DOTALL):
        def char_to_token(char_idx):
            for i, (start, end) in enumerate(offset_mapping):
                if end > char_idx:
                    return i
            return len(offset_mapping) - 1

        label = match.group(1).strip()
        
        char_start_full = char_offset + match.start(0)
        char_end_full = char_offset + match.end(0)
        char_start_content = char_offset + match.start(2)
        char_end_content = char_offset + match.end(2)

        token_start_full = char_to_token(char_start_full)
        token_end_full = char_to_token(char_end_full - 1)
        token_start_content = char_to_token(char_start_content)
        token_end_content = char_to_token(char_end_content - 1)

        if token_start_full <= token_end_full:
            boundaries.append({
                'label': label,
                'start_tag': (token_start_full, token_start_content),
                'content': (token_start_content, token_end_content + 1),
                'end_tag': (token_end_content + 1, token_end_full + 1),
                'full': (token_start_full, token_end_full + 1),
            })

    boundaries.sort(key=lambda b: b['full'][0])
    return boundaries

def get_full_attention(prompt: str, response: str, model_name: str, tokenizer: AutoTokenizer) -> tuple[np.ndarray, list[str], int]:
    """
    Loads a causal LM and calculates the full attention matrix.
    Returns: full_attention_matrix, all_tokens, prompt_token_len
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading causal model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
    model.to(device)
    model.eval()

    print("Processing input and calculating attention...")
    
    full_text = prompt + response
    prompt_inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    full_inputs = tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
    
    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
    
    prompt_token_len = prompt_inputs.input_ids.shape[1]
    all_tokens = tokenizer.convert_ids_to_tokens(full_inputs['input_ids'][0].cpu())

    with torch.no_grad():
        outputs = model(**full_inputs)

    last_layer_attentions = outputs.attentions[-1]
    full_attention_matrix = last_layer_attentions.squeeze(0).mean(dim=0).cpu().numpy()
    
    print("Applying strict causal mask...")
    seq_len = full_attention_matrix.shape[0]
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    full_attention_matrix[causal_mask] = 0
    
    print("Successfully extracted causal attention weights.")
    return full_attention_matrix, all_tokens, prompt_token_len

def segment_into_sentences(full_text: str, tokenizer: AutoTokenizer) -> list[dict]:
    """Segments text into sentences, treating <tags> as individual sentences."""
    encoding = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding.pop("offset_mapping").squeeze().tolist()
    
    def char_to_token(char_idx):
        for i, (start, end) in enumerate(offset_mapping):
            if end > char_idx: return i
        return len(offset_mapping) - 1

    char_segments = []
    special_token_pattern = r"<[^>]+>"
    last_char_end = 0

    for match in re.finditer(special_token_pattern, full_text):
        text_before = full_text[last_char_end:match.start()]
        if text_before.strip():
            sentence_ends = [m.end() for m in re.finditer(r'[.?!]', text_before)]
            sent_start = 0
            for end in sentence_ends:
                char_segments.append({'label': text_before[sent_start:end].strip(), 'start': last_char_end + sent_start, 'end': last_char_end + end})
                sent_start = end
            if sent_start < len(text_before):
                char_segments.append({'label': text_before[sent_start:].strip(), 'start': last_char_end + sent_start, 'end': match.start()})

        char_segments.append({'label': match.group(0), 'start': match.start(), 'end': match.end()})
        last_char_end = match.end()

    text_after = full_text[last_char_end:]
    if text_after.strip():
        sentence_ends = [m.end() for m in re.finditer(r'[.?!;:\n]', text_after)]
        sent_start = 0
        for end in sentence_ends:
            char_segments.append({'label': text_after[sent_start:end].strip(), 'start': last_char_end + sent_start, 'end': last_char_end + end})
            sent_start = end
        if sent_start < len(text_after):
            char_segments.append({'label': text_after[sent_start:].strip(), 'start': last_char_end + sent_start, 'end': len(full_text)})

    token_segments = []
    for seg in char_segments:
        if not seg['label']: continue
        token_start = char_to_token(seg['start'])
        token_end = char_to_token(seg['end'] - 1) + 1
        if token_start < token_end:
            token_segments.append({'label': seg['label'], 'slice': slice(token_start, token_end)})
    
    return token_segments

def save_attention_data(output_filepath: str, **kwargs):
    """Saves all provided data into a single compressed .npz file."""
    try:
        np.savez_compressed(output_filepath, **kwargs)
        print(f"\nSuccessfully saved attention data to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving data to file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model attention and save the data.")
    parser.add_argument('--config', type=str, default="config_cog.json", help='Path to the configuration JSON file.')
    args = parser.parse_args()

    # Load configuration from JSON file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    output_directory = config['output_directory']
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        print(f"Loading tokenizer: {config['tokenizer_name']}...")
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

        system_prompt = "You are a helpful assistant."
        if config.get('add_think_system_prompt', False):
            system_prompt += " You will always think before answer. Your thought should be wrapped in <think> and </think>. "

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": config['user_prompt']},
        ]
        input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        full_attn, all_toks, prompt_len = get_full_attention(
            prompt=input_prompt, 
            response=config['generated_response'],
            model_name=config['model_name'], 
            tokenizer=tokenizer
        )
        
        full_text = input_prompt + config['generated_response']
        
        node_bounds = find_node_boundaries(full_text, input_prompt, tokenizer)
        sentence_segs = segment_into_sentences(full_text, tokenizer)
        
        print("\n--- Analysis Results ---")
        print(f"Total Tokens: {len(all_toks)}")
        print(f"Found {len(node_bounds)} nodes: {[b['label'] for b in node_bounds]}")
        
        # Save all necessary data for the visualization script
        output_file = os.path.join(output_directory, "attention_data.npz")
        save_attention_data(
            output_filepath=output_file,
            attention_matrix=full_attn,
            all_tokens=np.array(all_toks, dtype=object),
            prompt_len=np.array(prompt_len),
            node_boundaries=np.array(node_bounds, dtype=object),
            sentence_segments=np.array(sentence_segs, dtype=object),
        )
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise
