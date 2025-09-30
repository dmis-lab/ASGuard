import os
import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class Model_HuggingFace:
    """
    Thin wrapper around Hugging Face causal language models used throughout this repository.
    This class is responsible for loading the base model, tokenizer and (optionally) caching a copy of the original weights for later ablation experiments. 
    To better support distributed training scenarios (e.g. multi‑GPU via Accelerate) the constructor accepts optional ``device`` and ``save_orig_state`` arguments.
    In the original implementation the model was always loaded onto the default CUDA device and the entire set of parameters was cloned into GPU memory via ``self.orig_state``, 
    which effectively doubles the memory footprint of large language models. 
    In a multi‑process setting this can lead to out‑of‑memory errors, since every process loads a full copy of the model on the same device (typically GPU 0).

    Parameters
    ----------
    model_name : str
        Key used to look up the Hugging Face model identifier in ``model_dict``.
    cache_dir : str, optional
        Directory used by Hugging Face to cache model files.
    save_orig_state : bool, default True
        If ``True``, clones and detaches each parameter from the loaded ``AutoModelForCausalLM`` into ``self.orig_state``. 
        This copy is used by ``get_response_ablation`` to restore the original weights after zeroing out specific attention heads.
        When set to ``False`` the copy is not created, saving a significant amount of GPU/CPU memory.
    device : torch.device or str, optional
        Device on which to load the model.  If ``None`` (the default), the constructor will use ``torch.device('cuda')`` if available, otherwise CPU.
        When running under ``accelerate``, you should pass ``accelerator.device`` so that each process loads its weights on the correct GPU.
    """

    def __init__(self, model_name, cache_dir, save_orig_state: bool = True, device: torch.device = None):
        model_dict = {
            "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
            "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
            "gemma2-9b": "google/gemma-2-9b-it",
        }
        self.system_prompts = {
            "llama3.1-8b": "You are a helpful AI assistant.",
            "qwen2.5-7b": "You are a helpful AI assistant.",
            "gemma2-9b": "",
        }
        
        # Determine device: default to CUDA if available, otherwise CPU
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.model_name = model_name
        
        # Load model on the specified device.  We pass a custom ``device_map`` that
        # assigns all model weights to the given device.  Later, ``accelerator.prepare``
        # may move parameters across devices if necessary.
        if "gemma" in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dict[model_name],
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True,
                cache_dir=cache_dir,
                attn_implementation="eager",
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dict[model_name],
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True,
                cache_dir=cache_dir,
            ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dict[model_name], token=os.getenv("HF_TOKEN"), cache_dir=cache_dir
        )
        
        # Optionally cache original weights.  We move the clone to CPU to free up GPU
        # memory.  When save_orig_state is False the attribute is set to None.
        if save_orig_state:
            self.orig_state = {
                k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()
            }
        else:
            self.orig_state = None

    def get_response(self, prompt, max_n_tokens, temperature):
        conv = [{"role": "user", "content": prompt}]
        if self.system_prompts[self.model_name] != "":
            conv = [{"role": "system", "content": self.system_prompts[self.model_name]}] + conv
        prompt_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_formatted, return_tensors='pt').to(self.device)

        outputs = self.model.generate(input_ids=inputs['input_ids'], max_new_tokens=max_n_tokens, temperature=temperature, do_sample=True)
        outputs_truncated = outputs[0][len(inputs['input_ids'][0]):]
        response = self.tokenizer.decode(outputs_truncated, skip_special_tokens=True)

        return response
    
    def get_response_ablation(
        self,
        prompt: str,
        max_n_tokens: int,
        temperature: float,
        temporal_head_masking: str = "none",
    ) -> str:
        # Restore original weights if available.  
        # Without a cached state dict we cannot perform ablation safely.
        if self.orig_state is None:
            raise ValueError(
                "orig_state is None. To use get_response_ablation, instantiate ModelHuggingFace with save_orig_state=True."
            )
        # Load the cached weights (stored on CPU) back into the model.  
        # This ensures a clean slate before zeroing out attention heads.
        self.model.load_state_dict(self.orig_state, strict=True)
        # Move model back to the desired device after loading CPU weights.
        self.model.to(self.device)
        print(temporal_head_masking)

        # Apply ablation by zeroing out the specified heads
        if temporal_head_masking != "none":
            for mask in temporal_head_masking.split(","):
                layer, head = map(int, re.match(r"a(\d+)h(\d+)", mask).groups())
                o_proj = self.model.model.layers[layer].self_attn.o_proj.weight.data
                # Unflatten the last dimension into (num_heads, head_dim)
                o_proj = o_proj.unflatten(
                    -1, (self.model.config.num_attention_heads, -1)
                )
                # Zero out the specified head
                o_proj[:, head, :] = 0

        # Generate a response using the modified model
        return self.get_response(prompt, max_n_tokens, temperature)

class ASGuard_Model(nn.Module):
    def __init__(
        self,
        model_name: str,
        target_heads_str: str,
        cache_dir: str = None,
        save_orig_state: bool = True,
        device: torch.device = None,
    ):
        super().__init__() 
        hf_wrapper = Model_HuggingFace(
            model_name, cache_dir, save_orig_state=save_orig_state, device=device
        )
        
        self.model = hf_wrapper.model
        self.model_name = hf_wrapper.model_name
        self.tokenizer = hf_wrapper.tokenizer 
        self.config = hf_wrapper.model.config
        self.device = hf_wrapper.device
        self.system_prompts = hf_wrapper.system_prompts
        
        self.get_response = hf_wrapper.get_response
        self.get_response_ablation = hf_wrapper.get_response_ablation
        
        model_dtype = self.model.dtype
        print(f"Base model loaded with dtype: {model_dtype}. Creating scaling vectors with the same dtype.")

        self.target_heads_map = self._parse_target_heads(target_heads_str)
        
        self.scaling_vectors = nn.ParameterDict()
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        for layer_idx, head_indices in self.target_heads_map.items():
            for head_idx in head_indices:
                key = f"layer{layer_idx}_head{head_idx}"
                self.scaling_vectors[key] = nn.Parameter(torch.ones(head_dim, dtype=model_dtype))
        
        self._hooks = []
        self._add_hooks()

    def _parse_target_heads(self, target_heads_str: str) -> dict:
        target_heads = {}
        if not target_heads_str:
            return target_heads
        for mask in target_heads_str.split(","):
            match = re.match(r"L(\d+)H(\d+)", mask)
            if match:
                layer, head = map(int, match.groups())
                if layer not in target_heads:
                    target_heads[layer] = []
                target_heads[layer].append(head)
        return target_heads

    def _create_hook(self, layer_idx: int, heads_to_scale: list):
        def hook(module, input, output):
            attention_output = output[0]
            attention_output = attention_output
            device = attention_output.device

            reshaped_output = attention_output.view(
                attention_output.size(0),
                attention_output.size(1),
                self.config.num_attention_heads,
                -1
            )
            
            for head_idx in heads_to_scale:
                key = f"layer{layer_idx}_head{head_idx}"
                if key in self.scaling_vectors:
                    scale = self.scaling_vectors[key].to(device)
                    reshaped_output[:, :, head_idx, :] *= scale
            
            scaled_output = reshaped_output.view(attention_output.size())
            return (scaled_output,) + output[1:]
        return hook

    def _add_hooks(self):
        for layer_idx, head_indices in self.target_heads_map.items():
            try:
                target_module = self.model.model.layers[layer_idx].self_attn
            except AttributeError:
                target_module = self.model.transformer.h[layer_idx].attn
            
            handle = target_module.register_forward_hook(self._create_hook(layer_idx, head_indices))
            self._hooks.append(handle)
            
    def remove_hooks(self):
        for h in getattr(self, "_hooks", []):
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_scaling_vectors(self, path: str):
        torch.save(self.scaling_vectors.state_dict(), path)
        print(f"Scaling vectors saved to {path}")

    def load_scaling_vectors(self, path: str):
        state = torch.load(path, map_location='cpu')
        target_dtype = next(self.scaling_vectors.parameters()).dtype
        state = {k: v.to(dtype=target_dtype) for k, v in state.items()}
        self.scaling_vectors.load_state_dict(state)
        print(f"Scaling vectors loaded from {path}")

    def merge_scales(self):
        self.remove_hooks() 

        num_heads = self.config.num_attention_heads
        head_dim  = self.config.hidden_size // num_heads

        for layer_idx, head_indices in self.target_heads_map.items():
            o_proj_weight = self.model.model.layers[layer_idx].self_attn.o_proj.weight.data
            reshaped = o_proj_weight.view(num_heads, head_dim, -1)

            for head_idx in head_indices:
                key = f"layer{layer_idx}_head{head_idx}"
                if key in self.scaling_vectors:
                    scale = self.scaling_vectors[key].detach().to(device=reshaped.device, dtype=reshaped.dtype)  # [head_dim]
                    reshaped[head_idx, :, :] *= scale.unsqueeze(1)                 # [head_dim,1] broadcast

            o_proj_weight.copy_(reshaped.view_as(o_proj_weight))

        print("Scaling vectors merged into o_proj. Hooks removed.")
