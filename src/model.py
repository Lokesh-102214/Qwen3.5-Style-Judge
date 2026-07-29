import os
import json
import torch
from unsloth import FastVisionModel
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from .config import MAX_SEQ_LEN

def fix_adapter_keys(adapter_path: str, fixed_dir: str = "fixed_adapter"):
    if not os.path.exists(fixed_dir):
        os.makedirs(fixed_dir)
        
    # If adapter_path is not a local folder, download it from HuggingFace
    if not os.path.exists(adapter_path):
        print(f"[MODEL] Downloading adapter from HF: {adapter_path}...")
        adapter_path = snapshot_download(repo_id=adapter_path)
    
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    fixed_safetensors = os.path.join(fixed_dir, "adapter_model.safetensors")
    
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_path}")
    
    print(f"[INFER] Remapping adapter keys...")
    state_dict = load_file(safetensors_path)
    new_state_dict = {
        key.replace("model.layers.", "model.language_model.layers."): value
        for key, value in state_dict.items()
    }
    save_file(new_state_dict, fixed_safetensors)
    
    with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
        
    # Patch adapter_config.json
    adapter_config["r"] = 16
    adapter_config["lora_alpha"] = 32
    
    with open(os.path.join(fixed_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
        
    return fixed_dir

def load_vision_model(base_model_name: str, adapter_path: str = None, load_in_4bit: bool = False):
    print(f"[MODEL] Loading Base Model: {base_model_name} …")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )
    
    FastVisionModel.for_inference(model)

    if adapter_path:
        fixed_adapter_dir = fix_adapter_keys(adapter_path)
        print(f"[INFER] Loading fixed adapter...")
        model.load_adapter(fixed_adapter_dir, adapter_name="default")
        
    print(f"[MODEL] Ready. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, tokenizer
