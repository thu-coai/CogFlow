# -*- coding: utf-8 -*-

"""
This script generates a JSON data file suitable for a web-based Sankey
visualizer, based on attention flow data from a saved .npz file.
It does NOT require transformers or the original model.

This script processes a token-level attention matrix by:
1.  Optionally applying token-level normalization (rank, log, power, or softmax).
2.  Aggregating attention scores between defined logical blocks (e.g., input, tags, content).
3.  Classifying the aggregated attention links into categories.
4.  Exporting a JSON file for each category, containing all nodes and links,
    with specific links highlighted based on category and optional top-k filtering.
    This JSON is designed to be consumed by a separate web application for visualization.
"""
import argparse
import os
import numpy as np
from scipy.stats import rankdata
import json


def build_and_merge_node_definitions(prompt_len, node_boundaries, all_tokens):
    """
    Builds a list of node segments and immediately merges all 'input' blocks into a single logical block.
    """
    # Step 1: Build all initial segments as before
    initial_segments = []
    prompt_start_index = 1 if all_tokens is not None and len(all_tokens) > 0 and all_tokens[0] == '<|begin_of_text|>' else 0
    initial_segments.append({'label': 'input', 'slice': slice(prompt_start_index, prompt_len)})
    last_idx = prompt_len

    for bound in node_boundaries:
        if bound['full'][0] > last_idx:
            inter_text_slice = slice(last_idx, bound['full'][0])
            if inter_text_slice.start < inter_text_slice.stop:
                initial_segments.append({'label': 'input', 'slice': inter_text_slice})

        start_tag_slice = slice(*bound['start_tag'])
        content_slice = slice(*bound['content'])
        end_tag_slice = slice(*bound['end_tag'])

        if start_tag_slice.start < start_tag_slice.stop:
            initial_segments.append({'label': f"{bound['label']}", 'slice': start_tag_slice})
        if content_slice.start < content_slice.stop:
            initial_segments.append({'label': f"{bound['label']}_content", 'slice': content_slice})
        if end_tag_slice.start < end_tag_slice.stop:
            initial_segments.append({'label': f"/{bound['label']}", 'slice': end_tag_slice})
        
        last_idx = bound['full'][1]

    if last_idx < len(all_tokens):
        final_text_slice = slice(last_idx, len(all_tokens))
        if final_text_slice.start < final_text_slice.stop:
            initial_segments.append({'label': 'final-text', 'slice': final_text_slice})

    # Step 2: Merge all 'input' slices
    merged_segments = []
    input_indices = []
    has_input = False
    
    for seg in initial_segments:
        if seg['label'] == 'input':
            input_indices.extend(range(seg['slice'].start, seg['slice'].stop))
            has_input = True
        else:
            # For non-input blocks, we directly store their index list
            seg['indices'] = list(range(seg['slice'].start, seg['slice'].stop))
            merged_segments.append(seg)

    # If input blocks existed, create a unified input block and place it at the beginning of the list
    if has_input:
        # Use a set to ensure unique indices, then sort
        unique_input_indices = sorted(list(set(input_indices)))
        unified_input_segment = {'label': 'input', 'indices': unique_input_indices}
        # Insert the unified input block at the start of the final list
        merged_segments.insert(0, unified_input_segment)
        
    return merged_segments


def normalize_attention_matrix(matrix, method='none', temperature=1.0, power=1.0):
    """
    Normalizes the token-level attention matrix before aggregation.
    """
    if method == 'none':
        return matrix

    norm_matrix = np.copy(matrix).astype(float)
    num_tokens = matrix.shape[0]

    if method == 'rank':
        print("Applying 'rank' normalization to token-level attention...")
        for s in range(1, num_tokens):
            attention_slice = norm_matrix[s, :s]
            positive_mask = attention_slice > 0
            if not np.any(positive_mask): continue
            values_to_rank = attention_slice[positive_mask]
            ranks = rankdata(values_to_rank, method='dense')
            max_rank = ranks.max()
            normalized_ranks = (ranks - 1) / (max_rank - 1) if max_rank > 1 else np.full_like(ranks, 0.5, dtype=float)
            attention_slice[positive_mask] = normalized_ranks
            norm_matrix[s, :s] = attention_slice

    elif method == 'log':
        print("Applying 'log' normalization to token-level attention...")
        norm_matrix = np.log1p(norm_matrix)
        min_val, max_val = norm_matrix.min(), norm_matrix.max()
        norm_matrix = (norm_matrix - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(norm_matrix)

    elif method == 'power':
        print(f"Applying 'power' normalization with exponent={power}...")
        positive_mask = norm_matrix > 0
        norm_matrix[positive_mask] = np.power(norm_matrix[positive_mask], power)
        min_val, max_val = norm_matrix.min(), norm_matrix.max()
        norm_matrix = (norm_matrix - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(norm_matrix)

    elif method == 'softmax':
        print(f"Applying 'softmax' normalization with temperature={temperature}...")
        for s in range(1, num_tokens):
            attention_slice = norm_matrix[s, :s]
            positive_mask = attention_slice > 0
            if not np.any(positive_mask): continue
            values_to_softmax = attention_slice[positive_mask]
            scaled_values = values_to_softmax / temperature
            exp_values = np.exp(scaled_values - np.max(scaled_values))
            softmax_values = exp_values / np.sum(exp_values)
            attention_slice[positive_mask] = softmax_values
            norm_matrix[s, :s] = attention_slice

    return norm_matrix

def normalize_1d_array(array, method='none', power=1.0):
    """Normalizes a 1D array for visualization scaling."""
    if method == 'none' or array.size == 0:
        min_val, max_val = array.min(), array.max()
        if max_val > min_val:
            return (array - min_val) / (max_val - min_val + 1e-9)
        return np.zeros_like(array)

    norm_array = np.copy(array).astype(float)

    if method == 'rank':
        ranks = rankdata(norm_array, method='dense')
        max_rank = ranks.max()
        return (ranks - 1) / (max_rank - 1) if max_rank > 1 else np.full_like(ranks, 0.5, dtype=float)

    elif method == 'log':
        norm_array = np.log1p(norm_array)
    elif method == 'power':
        norm_array = np.power(norm_array, power)

    min_val, max_val = norm_array.min(), norm_array.max()
    if max_val > min_val:
        return (norm_array - min_val) / (max_val - min_val + 1e-9)
    return np.zeros_like(array)

def calculate_graph_data(data, attention_matrix, args):
    """
    Calculates aggregated attention data without performing visualization.
    Returns node definitions, node info, and all calculated edges.
    """
    prompt_len, all_tokens = int(data['prompt_len']), data.get('all_tokens')
    # Use the new merging function
    segments = build_and_merge_node_definitions(prompt_len, data['node_boundaries'], all_tokens)
    
    if not segments:
        print("Warning: All node segments were empty after merge.")
        return None, None, None

    agg_matrix = np.zeros((len(segments), len(segments)))
    for i, query_seg in enumerate(segments):
        for j, key_seg in enumerate(segments):
            # Only calculate attention from previous blocks to the current block
            if j >= i: continue

            query_indices = query_seg['indices']
            key_indices = key_seg['indices']
            
            if not query_indices or not key_indices: continue

            # Create sub-matrix from index lists using np.ix_
            sub_matrix = attention_matrix[np.ix_(query_indices, key_indices)]
            if sub_matrix.size == 0: continue

            if args.agg_method == "mean":
                agg_matrix[i, j] = sub_matrix.mean()

            elif args.agg_method == "max_mean":
                agg_matrix[i, j] = np.mean(np.max(sub_matrix, axis=1))

            elif args.agg_method == "mean_max":
                agg_matrix[i, j] = np.max(np.mean(sub_matrix, axis=0))

            elif args.agg_method == "max_max":
                agg_matrix[i, j] = np.max(sub_matrix)

            elif args.agg_method == "topk_mean":
                scores_list = [np.mean(row) for row in sub_matrix if row.size > 0]
                if scores_list:
                    scores = np.array(scores_list)
                    k_val = min(args.agg_top_k, len(scores))
                    agg_matrix[i, j] = np.mean(np.sort(scores)[-k_val:])

            elif args.agg_method == "topk_topk_mean":
                affectedness_scores = []
                for k_idx in range(sub_matrix.shape[0]):
                    attention_to_key = sub_matrix[k_idx, :]
                    if attention_to_key.size > 0:
                        sorted_attentions = np.sort(attention_to_key)[::-1]
                        num_to_take = min(args.agg_inner_top_k, len(sorted_attentions))
                        top_k_attentions = sorted_attentions[:num_to_take]
                        affectedness_scores.append(np.mean(top_k_attentions))
                if affectedness_scores:
                    scores = np.array(affectedness_scores)
                    k_val = min(args.agg_top_k, len(scores))
                    agg_matrix[i, j] = np.mean(np.sort(scores)[-k_val:])
    
    node_ids = [f'n{i}' for i in range(len(segments))]
    node_info_list = []
    for i, seg in enumerate(segments):
        is_content = seg['label'].endswith('_content')
        font_name = "Helvetica"
        font_color = "#2c3e50"
        if is_content:
            style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#fdebd0', 'penwidth': '0', 'fontname': font_name, 'fontcolor': font_color}
        elif seg['label'] in ['input', 'final-text']:
            style = {'shape': 'box', 'style': 'filled', 'fillcolor': '#ecf0f1', 'penwidth': '0', 'fontname': font_name, 'fontcolor': font_color}
        else:
            style = {'shape': 'box', 'style': 'filled', 'fillcolor': '#d4e6f1', 'penwidth': '0', 'fontname': font_name, 'fontcolor': font_color}
        node_full_label = seg['label']
        node_info_list.append({'id': node_ids[i], 'label': node_full_label, **style})

    all_edges = [{'source': node_ids[j], 'target': node_ids[i], 'weight': agg_matrix[i, j]}
                 for i in range(len(segments)) for j in range(i)
                 if agg_matrix[i, j] > 0]

    return segments, node_info_list, all_edges


def get_node_type(label):
    """Returns the type of a node ('input', 'content', 'tag', 'response') based on its label."""
    if label == 'input':
        return 'input'
    if label == 'final-text':
        return 'response'
    if label.endswith('_content'):
        return 'content'
    # All other labels are considered 'tag'
    return 'tag'

def classify_edges(nodes, edges):
    """Classifies edges based on new criteria."""
    node_label_map = {node['id']: node['label'] for node in nodes}
    
    category1_edges = []
    category2_edges = []
    category3_edges = []

    for edge in edges:
        source_label = node_label_map.get(edge['source'], 'Unknown')
        target_label = node_label_map.get(edge['target'], 'Unknown')
        
        source_type = get_node_type(source_label)
        target_type = get_node_type(target_label)

        # Category 1: input->tag, tag->tag, tag->response
        if (source_type == 'input' and target_type == 'tag') or \
           (source_type == 'tag' and target_type == 'tag') or \
           (source_type == 'tag' and target_type == 'response'):
            category1_edges.append(edge)

        # Category 2: tag->content, content->tag
        if (source_type == 'tag' and target_type == 'content') or \
           (source_type == 'content' and target_type == 'tag'):
            category2_edges.append(edge)

        # Category 3: input->content, content->content, content->response
        if (source_type == 'input' and target_type == 'content') or \
           (source_type == 'content' and target_type == 'content') or \
           (source_type == 'content' and target_type == 'response'):
            category3_edges.append(edge)
            
    return {
        "category1": category1_edges,
        "category2": category2_edges,
        "category3": category3_edges,
    }

def export_sankey_data_for_web(node_info_list, all_edges, highlighted_edges, args, output_path):
    """
    Exports graph data to a JSON format for an interactive web app.
    All edges are included, but only highlighted edges have full opacity.
    """
    # Define your mapping rules here, identical to the ones in create_sankey_with_plotly
    SANKEY_LABEL_MAP = {
        "input": "Input",
        "Observation": "<Observation>",
        "Observation_content": "Content_obser.",
        "/Observation": "</Observation>",
        
        "Regulation": "<Regulation>",
        "Regulation_content": "Content_regu.",
        "/Regulation": "</Regulation>",
        
        "Attribution": "<Attribution>",
        "Attribution_content": "Content_attr.",
        "/Attribution": "</Attribution>",
        
        "Motivation": "<Motivation>",
        "Motivation_content": "Content_motiv.",
        "/Motivation": "</Motivation>",
        
        "Efficacy": "<Efficacy>",
        "Efficacy_content": "Content_eff.",
        "/Efficacy": "</Efficacy>",
        
        "final-text": "Response",
    }

    # Use the mapping dictionary to get the desired node labels for the JSON output
    web_nodes = [{'id': node['id'], 'label': SANKEY_LABEL_MAP.get(node['label'], node['label']), 'color': node['fillcolor']} for node in node_info_list]
    
    # Create a set of highlighted edges for efficient lookup
    highlighted_set = set((e['source'], e['target']) for e in highlighted_edges)

    # Normalize weights based on all edges for a consistent visual scale
    all_weights_values = np.array([edge['weight'] for edge in all_edges] if all_edges else [0])
    norm_weights = normalize_1d_array(all_weights_values, method=args.viz_norm_method, power=args.viz_power)

    color_map_hex = {'content': '#e74c3c', 'tag': '#3498db', 'other': '#95a5a6'}
    id_to_label = {info['id']: info['label'] for info in node_info_list}
    
    # Define alpha ranges for highlighted and faded links
    highlighted_min_alpha, highlighted_max_alpha = 0.4, 0.9
    faded_min_alpha, faded_max_alpha = 0.05, 0.15

    web_links = []
    for i, edge in enumerate(all_edges):
        source_label_full = id_to_label.get(edge['source'], '')
        base_hex_color = color_map_hex['other']
        if source_label_full.endswith('_content'):
            base_hex_color = color_map_hex['content']
        elif source_label_full not in ['input', 'final-text']:
            base_hex_color = color_map_hex['tag']
        
        # --- NEW OPACITY LOGIC ---
        is_highlighted = (edge['source'], edge['target']) in highlighted_set
        
        normalized_weight = norm_weights[i] if len(norm_weights) > i else 0
        width = 1 + normalized_weight * 15

        if is_highlighted:
            # Scale opacity within the highlighted range
            final_alpha = highlighted_min_alpha + normalized_weight * (highlighted_max_alpha - highlighted_min_alpha)
        else:
            # Scale opacity within the faded range
            final_alpha = faded_min_alpha + normalized_weight * (faded_max_alpha - faded_min_alpha)

        web_links.append({
            "source": edge['source'],
            "target": edge['target'],
            "value": edge['weight'],
            "style": {
                "color": base_hex_color,
                "opacity": round(final_alpha, 3),
                "width": round(width, 2)
            }
        })
    
    web_data = {"nodes": web_nodes, "links": web_links}
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(web_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully exported styled data for web app to: {output_path}")
    except Exception as e:
        print(f"Error exporting styled data for web app: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sankey JSON data from an attention flow .npz file.")
    parser.add_argument("data_file", type=str, help="Name of the attention_data.npz file.")
    parser.add_argument("--run_folder", type=str, default="attention_maps_cog_4", help="Unified folder for input data file and all output files.")
    parser.add_argument("-k", "--top_k", type=int, default=None, help="Highlight only the top-k incoming attention edges for each node in the Sankey JSON.")
    parser.add_argument("--agg_method", type=str, default="topk_topk_mean", choices=["mean", "max_mean", "mean_max", "max_max", "topk_mean", "topk_topk_mean"], help="Aggregation method for attention scores.")
    parser.add_argument("--agg_top_k", type=int, default=3, help="Value of outer 'k' for 'topk_mean' and 'topk_topk_mean' aggregation.")
    parser.add_argument("--agg_inner_top_k", type=int, default=3, help="Value of inner 'k' for 'topk_topk_mean' aggregation.")
    parser.add_argument("--norm_method", type=str, default="none", choices=["none", "rank", "log", "softmax", "power"], help="Token-level attention normalization method before aggregation.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax normalization. >1 for less concentration, <1 for more.")
    parser.add_argument("--power", type=float, default=1.0, help="Exponent for power normalization. <1 for less concentration, >1 for more.")
    parser.add_argument("--viz_norm_method", type=str, default="none", choices=["none", "rank", "log", "power"], help="Normalization method for final visualization scaling in the JSON output.")
    parser.add_argument("--viz_power", type=float, default=1.0, help="Exponent for 'power' visualization normalization.")
    args = parser.parse_args()

    output_dir = args.run_folder
    os.makedirs(output_dir, exist_ok=True)

    data_file_path = os.path.join(output_dir, args.data_file)
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}"); exit(1)

    print(f"Loading data from {data_file_path}...")
    data = np.load(data_file_path, allow_pickle=True)

    normalized_attention_matrix = normalize_attention_matrix(data['attention_matrix'], method=args.norm_method, temperature=args.temperature, power=args.power)

    suffix = ""
    if args.agg_method != "mean": suffix += f"_{args.agg_method}"
    if args.agg_method == "topk_mean": suffix += f"_k{args.agg_top_k}"
    if args.agg_method == "topk_topk_mean": suffix += f"_k{args.agg_top_k}-ik{args.agg_inner_top_k}"
    if args.norm_method != "none":
        suffix += f"_{args.norm_method}"
        if args.norm_method == 'softmax': suffix += f"_t{args.temperature}"
        if args.norm_method == 'power': suffix += f"_p{args.power}"
    if args.top_k: suffix += f"_top{args.top_k}"
    if args.viz_norm_method != "none":
        suffix += f"_viz-{args.viz_norm_method}"
        if args.viz_norm_method == 'power': suffix += f"p{args.viz_power}"

    print("\n--- Calculating Aggregated Attention Graph Data ---")
    segments, node_list, all_edge_list = calculate_graph_data(data, attention_matrix=normalized_attention_matrix, args=args)
    
    if segments and node_list:
        # --- Generate and Export Sankey JSON Data ---
        if all_edge_list:
            print("\n--- Creating Sankey-Style Attention Flow JSON Data ---")
            
            categorized_edges = classify_edges(node_list, all_edge_list)
            
            sankey_categories = {
                "category1_IT_TT_TR": categorized_edges["category1"],
                "category2_TC_CT": categorized_edges["category2"],
                "category3_IC_CC_CR": categorized_edges["category3"],
            }

            for category_name, category_specific_edges in sankey_categories.items():
                edges_to_highlight = category_specific_edges
                # Optionally apply top_k to the set of highlighted edges
                if args.top_k:
                    filtered_highlight_edges = []
                    node_ids = [n['id'] for n in node_list]
                    for node_id in node_ids:
                        incoming = sorted([e for e in category_specific_edges if e['target'] == node_id], key=lambda x: x['weight'], reverse=True)
                        filtered_highlight_edges.extend(incoming[:args.top_k])
                    edges_to_highlight = filtered_highlight_edges

                if not all_edge_list:
                    print(f"No edges exist to generate a Sankey JSON for '{category_name}'. Skipping.")
                    continue
                
                base_output_path = os.path.join(output_dir, f"attention_sankey_{category_name}{suffix}")
                sankey_json_path = f"{base_output_path}.json"
                
                # Generate the Web App JSON file
                export_sankey_data_for_web(node_list, all_edge_list, edges_to_highlight, args, sankey_json_path)
        else:
            print("No edges were calculated. Skipping JSON export.")
    else:
        print("Could not calculate graph data (no segments or nodes). Skipping JSON export.")
