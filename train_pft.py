import os
import argparse
import json
import logging
import random
from typing import Dict, List, Tuple
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForCausalLM

from ASGuard.model import ASGuard_Model

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import enable_fsdp_ram_efficient_loading, disable_fsdp_ram_efficient_loading
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import bitsandbytes as bnb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SteeringDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def _collect_transformer_layers() -> Tuple[type, ...]:
    layer_types = []
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        layer_types.append(LlamaDecoderLayer)
    except Exception: pass
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        layer_types.append(Qwen2DecoderLayer)
    except Exception: pass
    try:
        from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
        layer_types.append(Gemma2DecoderLayer)
    except Exception: pass
    try:
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
        layer_types.append(MistralDecoderLayer)
    except Exception: pass
    try:
        from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
        layer_types.append(Phi3DecoderLayer)
    except Exception: pass
    return tuple(layer_types)

def _build_fsdp_plugin() -> FullyShardedDataParallelPlugin:
    transformer_layer_cls = _collect_transformer_layers()
    auto_policy = None
    if transformer_layer_cls:
        auto_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls
        )

    base_kwargs = dict(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,          
        limit_all_gathers=True,
        cpu_offload=False,
        auto_wrap_policy=auto_policy,
    )

    attempts = [
        dict(activation_checkpointing=transformer_layer_cls, forward_prefetch=True, backward_prefetch=BackwardPrefetch.BACKWARD_PRE),
        dict(activation_checkpointing=transformer_layer_cls, forward_prefetch=True),
        dict(activation_checkpointing=transformer_layer_cls),
        dict(),
    ]
    last_err = None
    for extra in attempts:
        try:
            return FullyShardedDataParallelPlugin(**base_kwargs, **extra)
        except TypeError as e:
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError("FSDP plugin creation failed")


def main():
    parser = argparse.ArgumentParser(description="Finetune a model with Preventative Scaling Vector.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--target_heads", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1) 
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
    )
    
    logger.info(f"Loading dataset from {args.data_path}")
    df = pd.read_csv(args.data_path)
    df.dropna(subset=['request_reformulated'], inplace=True)
    
    df = df[df['request_reformulated'].apply(lambda x: isinstance(x, str) and x.strip() != '')]
    logger.info(f"Loaded and cleaned dataset. Total samples: {len(df)}")
    safe_refusal_response = "I'm sorry, but I cannot fulfill that request."
    train_data = [{'input': row['request_reformulated'], 'output': safe_refusal_response} for _, row in df.iterrows()]
    train_dataset = SteeringDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    logger.info("Instantiating ASGuard model for preventative steering...")
    enable_fsdp_ram_efficient_loading()
    sv_model = ASGuard_Model(
        args.model_name,
        args.target_heads,
        cache_dir=args.cache_dir,
        save_orig_state=False,
    )
    disable_fsdp_ram_efficient_loading()
    sv_model.model.gradient_checkpointing_enable()
    sv_model.model.config.use_cache = False

    logger.info(f"Loading original scaling vector from {args.sv_vector_path}")
    sv_model.load_scaling_vectors(args.sv_vector_path)
    
    for p in sv_model.scaling_vectors.parameters():
        p.requires_grad_(False)
    
    tokenizer = sv_model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    optimizer = bnb.optim.AdamW8bit(
        (p for p in sv_model.model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.05
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.epochs
    )
    
    sv_model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        sv_model, optimizer, train_dataloader, scheduler
    )
    
    logger.info("===== Starting Preventative Finetuning =====")
    sv_model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            with accelerator.accumulate(sv_model):
                optimizer.zero_grad()
                
                batch_size = len(batch['input'])
                full_texts = []
                for i in range(batch_size):
                    user_input = batch['input'][i]
                    target_output = batch['output'][i]
                    formatted_text = tokenizer.apply_chat_template([{"role": "user", "content": user_input}], tokenize=False) + target_output + tokenizer.eos_token
                    full_texts.append(formatted_text)
                
                inputs = tokenizer(full_texts, return_tensors='pt', padding="longest", truncation=True, max_length=args.max_length).to(accelerator.device)
                labels = inputs.input_ids.clone()
                
                for i in range(batch_size):
                    prompt_only = tokenizer.apply_chat_template([{"role": "user", "content": batch['input'][i]}], tokenize=False)
                    prompt_tokens_len = len(tokenizer(prompt_only, add_special_tokens=False)['input_ids'])
                    labels[i, :prompt_tokens_len] = -100

                outputs = sv_model(inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
                loss = outputs.loss
                
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f"Saving final model to {args.output_dir}")
        unwrapped_model = accelerator.unwrap_model(sv_model)
        unwrapped_model.remove_hooks()
        unwrapped_model.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        config_data = vars(args)
        with open(os.path.join(args.output_dir, "training_config.json"), 'w') as f:
            json.dump(config_data, f, indent=4)
            
    accelerator.print("Preventative finetuning completed successfully!")

if __name__ == "__main__":
    main()