import argparse
import json
import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from ASGuard.model import ASGuard_Model, Model_HuggingFace

def prepare_probe_dataset(csv_path: str, num_samples: int = 100):
    print(f"--- Preparing probe dataset from {csv_path} ---")
    df = pd.read_csv(csv_path)
    present_tense_prompts = df['request'].tolist()
    past_tense_prompts = df['request_reformulated'].tolist()

    if num_samples > 0:
        return present_tense_prompts[:num_samples], past_tense_prompts[:num_samples]
    return present_tense_prompts, past_tense_prompts

def prepare_probe_dataset_category(csv_path: str, num_samples: int = -1):
    print(f"--- Preparing probe dataset from {csv_path} with category filtering ---")
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['request', 'request_reformulated', 'category']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: The CSV file must contain the columns: {required_columns}")
            exit()
            
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        exit()

    original_count = len(df)
    df_filtered = df[df['category'] == 'false_to_true'].copy()
    filtered_count = len(df_filtered)
    print(f"Filtered for 'false_to_true' category: {original_count} -> {filtered_count} samples.")

    if filtered_count == 0:
        print("Warning: No samples found with category 'false_to_true'. Cannot perform analysis.")
        return [], []

    if num_samples > 0:
        df_filtered = df_filtered.head(num_samples)

    present_tense_prompts = df_filtered['request'].tolist()
    past_tense_prompts = df_filtered['request_reformulated'].tolist()
    
    print(f"Final analysis dataset created with {len(present_tense_prompts)} samples.")
    return present_tense_prompts, past_tense_prompts

def get_activations(model_wrapper: Model_HuggingFace, prompts: list, layer_idx: int):
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    device = model_wrapper.device
    
    activations = []
    
    def hook_fn(module, input, output):
        last_token_activation = input[0][:, -1, :].detach().cpu()
        activations.append(last_token_activation)

    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(hook_fn)

    print(f"Extracting activations from layer {layer_idx}...")
    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Getting Activations L{layer_idx}"):
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256).to(device)
            model(**inputs)

    hook_handle.remove()
    return torch.cat(activations)

def get_activations_inocul(model, tokenizer, device, prompts: list, layer_idx: int):
    model = model
    tokenizer = tokenizer
    device = device
    
    activations = []
    
    def hook_fn(module, input, output):
        last_token_activation = input[0][:, -1, :].detach().cpu()
        activations.append(last_token_activation)

    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(hook_fn)

    print(f"Extracting activations from layer {layer_idx}...")
    with torch.no_grad():
        for prompt in tqdm(prompts, desc=f"Getting Activations L{layer_idx}"):
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256).to(device)
            model(**inputs)

    hook_handle.remove()
    return torch.cat(activations)

def plot_single_head_distribution(past_scores, present_scores, layer, head, output_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(past_scores, color="skyblue", label="Past Tense Prompts", kde=True, stat="density")
    sns.histplot(present_scores, color="red", label="Present Tense Prompts", kde=True, stat="density", alpha=0.6)
    plt.title(f"Probe Analysis: Dot Product Scores for L{layer}H{head}")
    plt.xlabel("Dot Product Score (Activation · Probe Vector)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_summary_accuracy(results: dict, output_path: str):
    head_labels = list(results.keys())
    accuracies = [res['accuracy'] for res in results.values()]

    plt.figure(figsize=(8, max(6, len(head_labels) * 0.5)))
    bars = sns.barplot(x=accuracies, y=head_labels, orient='h', palette="viridis")
    
    for i, v in enumerate(accuracies):
        bars.text(0.02, i, f"{v:.2%}", color='black', va='center', ha='left', fontweight='bold')

    plt.axvline(x=0.5, color='r', linestyle='--', label='Random Chance (50%)')
    plt.title("Summary of Linear Probe Classification Accuracy per Head")
    plt.xlabel("Classification Accuracy")
    plt.ylabel("Attention Head (Layer, Head)")
    plt.xlim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n--- 📊 Summary plot saved to {output_path} ---")
    plt.close()


def main(args):
    config_path = os.path.join(args.sv_path, "scaling_vector_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_name = config['model_name']
    
    target_heads_by_layer = defaultdict(list)
    for head_str in args.target_heads.split(','):
        match = re.match(r"L(\d+)H(\d+)", head_str)
        if match:
            layer, head = map(int, match.groups())
            target_heads_by_layer[layer].append(head)

    print("--- Loading ASGuard Scaled model to extract all probe vectors ---")
    sv_model_wrapper = ASGuard_Model(model_name, args.target_heads)
    pt_files = [f for f in os.listdir(args.sv_path) if f.endswith('.pt')]
    scales_path = os.path.join(args.sv_path, pt_files[0])
    sv_model_wrapper.load_scaling_vectors(scales_path)
    
    probe_vectors = {}
    for layer, heads in target_heads_by_layer.items():
        for head in heads:
            key = f"layer{layer}_head{head}"
            if key in sv_model_wrapper.scaling_vectors:
                probe_vectors[key] = sv_model_wrapper.scaling_vectors[key].detach()
            else:
                print(f"Warning: L{layer}H{head} not found in ASGuard model, skipping.")
    
    del sv_model_wrapper
    torch.cuda.empty_cache()

    print("\n--- Loading BASE model to extract activations ---")
    
    if args.inocul_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"--- Loading inoculated model: {args.inocul_path} ---")
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(args.inocul_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.inocul_path)
    else:
        baseline_model_wrapper = Model_HuggingFace(model_name)
    present_prompts, past_prompts = prepare_probe_dataset_category(args.dataset_path, num_samples=args.num_samples)
    
    activations_by_layer = {}
    if args.inocul_path:
        for layer_idx in target_heads_by_layer.keys():
            past_activations = get_activations_inocul(model, tokenizer, device, past_prompts, layer_idx)
            present_activations = get_activations_inocul(model, tokenizer, device, present_prompts, layer_idx)
            activations_by_layer[layer_idx] = (past_activations, present_activations)
            
        del model
    else:
        for layer_idx in target_heads_by_layer.keys():
            past_activations = get_activations(baseline_model_wrapper, past_prompts, layer_idx)
            present_activations = get_activations(baseline_model_wrapper, present_prompts, layer_idx)
            activations_by_layer[layer_idx] = (past_activations, present_activations)
        
        del baseline_model_wrapper
    torch.cuda.empty_cache()

    all_results = {}
    analysis_output_dir = os.path.join(args.sv_path, "probe_analysis")
    os.makedirs(analysis_output_dir, exist_ok=True)

    print("\n--- 🔬 Performing linear probe analysis for each head ---")
    for layer, heads in tqdm(target_heads_by_layer.items(), desc="Analyzing Heads"):
        past_activations, present_activations = activations_by_layer[layer]
        
        for head in heads:
            head_label = f"L{layer}H{head}"
            probe_vector_key = f"layer{layer}_head{head}"
            if probe_vector_key not in probe_vectors:
                continue

            probe_vector = probe_vectors[probe_vector_key]
            
            head_dim = probe_vector.shape[0]
            num_heads = past_activations.shape[-1] // head_dim
            
            past_head_activations = past_activations.view(-1, num_heads, head_dim)[:, head, :]
            present_head_activations = present_activations.view(-1, num_heads, head_dim)[:, head, :]
            
            probe_vector_converted = probe_vector.cpu().to(past_head_activations.dtype)
            
            past_scores = (past_head_activations @ probe_vector_converted).float().numpy()
            present_scores = (present_head_activations @ probe_vector_converted).float().numpy()
            
            threshold = (np.mean(past_scores) + np.mean(present_scores)) / 2
            correct = np.sum(past_scores > threshold) + np.sum(present_scores <= threshold)
            accuracy = correct / (len(past_scores) + len(present_scores))
            
            all_results[head_label] = {'accuracy': accuracy}
            print(f"  - {head_label} Accuracy: {accuracy:.2%}")
            
            if args.inocul_path:
                plot_path = os.path.join(args.inocul_path, f"distribution_{head_label}.png")
            else: 
                plot_path = os.path.join(analysis_output_dir, f"distribution_{head_label}.png")
            plot_single_head_distribution(past_scores, present_scores, layer, head, plot_path)
            
    if args.inocul_path:
        summary_plot_path = os.path.join(args.inocul_path,"probe_accuracy_summary.png")
    else:
        summary_plot_path = os.path.join(args.sv_path, "probe_accuracy_summary.png")
    plot_summary_accuracy(all_results, summary_plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze multiple scaling vectors as linear probes.")
    parser.add_argument("--sv_path", type=str, required=True, help="Directory of the trained scaling vector.")
    parser.add_argument("--inocul_path", type=str, default=None, help="Path to the directory containing trained inoculated model")
    parser.add_argument("--target_heads", type=str, required=True, help="Comma-separated string of heads to analyze (e.g., 'L13H8,L14H14').")
    parser.add_argument("--dataset_path", type=str, default="classification.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to use for analysis.")
    args = parser.parse_args()
    main(args)