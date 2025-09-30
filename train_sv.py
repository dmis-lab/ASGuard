import os
import torch
import pandas as pd
import argparse
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import json

from ASGuard.model import ASGuard_Model

def create_dataset_from_csv(csv_path: str, num_samples: int = -1):
    print(f"Loading and preprocessing dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        if 'request_reformulated' not in df.columns:
            print("Error: The CSV file must contain a 'request_reformulated' column.")
            exit()
            
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        print("Please provide the correct path to the dataset file.")
        exit()

    if num_samples > 0:
        df = df.head(num_samples)

    ideal_refusal = "I'm sorry, but I cannot fulfill that request."

    dataset = []
    for prompt in tqdm(df['request_reformulated'], desc="Loading Prompts from CSV"):
        if not isinstance(prompt, str):
            continue
        dataset.append({
            "prompt": prompt,
            "target_response": ideal_refusal
        })
        
    print(f"Dataset created with {len(dataset)} samples.")
    return dataset

def main(args):
    asguard_model = ASGuard_Model(
        model_name=args.model_name,
        target_heads_str=args.target_heads
    )
    asguard_model.model.to(asguard_model.device)
    
    dataset = create_dataset_from_csv(args.dataset_path, num_samples=args.num_samples)

    optimizer = AdamW(asguard_model.scaling_vectors.parameters(), lr=args.learning_rate)
    
    print("\nStarting Scaling Vector training...")
    
    asguard_model.model.train() 
    
    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for data in progress_bar:
            prompt = data["prompt"]
            target_response = data["target_response"]

            conversation = []
            system_prompt = asguard_model.system_prompts.get(args.model_name, "")

            if 'gemma' in args.model_name.lower():
                full_user_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                conversation = [
                    {"role": "user", "content": full_user_prompt},
                    {"role": "assistant", "content": target_response}
                ]
            else:
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target_response}
                ]

            full_text = asguard_model.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            ) + asguard_model.tokenizer.eos_token

            inputs = asguard_model.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512).to(asguard_model.device)
            input_ids = inputs.input_ids
            
            assistant_response_ids = asguard_model.tokenizer(
                target_response + asguard_model.tokenizer.eos_token,
                return_tensors='pt',
                add_special_tokens=False
            ).input_ids

            labels = input_ids.clone()
            prompt_len = input_ids.shape[1] - assistant_response_ids.shape[1]
            labels[:, :prompt_len] = -100 

            optimizer.zero_grad()

            outputs = asguard_model.model(input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    scales_filename = f"asguard_scales_{args.model_name}.pt"
    scales_path = os.path.join(args.output_dir, scales_filename)
    asguard_model.save_scaling_vectors(scales_path)

    config_data = {
        "model_name": args.model_name,
        "target_heads": args.target_heads,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs
    }
    config_filename = "scaling_vector_config.json"
    config_path = os.path.join(args.output_dir, config_filename)
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    
    print(f"\nTraining finished.")
    print(f"Scaling vectors saved to: {scales_path}")
    print(f"Training config saved to: {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model to mitigate targeted jailbreaking using a predefined CSV dataset.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Hugging Face model to train (e.g., 'llama3.1-8b').")
    parser.add_argument("--target_heads", type=str, required=True, help="Comma-separated string of target heads (e.g., 'L0H3,L10H19').")
    parser.add_argument("--dataset_path", type=str, default="classification.csv", help="Path to the training dataset CSV file.")
    parser.add_argument("--output_dir", type=str, default="asguard_checkpoints", help="Directory to save the trained scaling vectors.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to use from the dataset (-1 for all).")
    
    args = parser.parse_args()
    main(args)