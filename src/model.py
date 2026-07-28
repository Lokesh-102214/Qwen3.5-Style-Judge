import torch
from unsloth import FastVisionModel
from .config import MAX_SEQ_LEN

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
        print(f"[MODEL] Loading Adapter: {adapter_path} …")
        model.load_adapter(adapter_path, adapter_name="default")
        
    print(f"[MODEL] Ready. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, tokenizer
