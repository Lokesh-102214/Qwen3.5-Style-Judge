import re
import torch
from typing import Dict
from qwen_vl_utils import process_vision_info
from .config import SYSTEM_PROMPT, MAX_NEW_TOKENS

def extract_multi_metrics(llm_output: str) -> Dict[str, any]:
    """Parse 3 integer scores from XML output and compute the final scaled score."""
    raw = {"sf": 0, "cp": 0, "rq": 0}
    
    sf_m = re.search(r'Style Fidelity:\s*([0-9]+)', llm_output, re.I)
    cp_m = re.search(r'Content Preservation:\s*([0-9]+)', llm_output, re.I)
    rq_m = re.search(r'Rendering Quality:\s*([0-9]+)', llm_output, re.I)
    
    raw["sf"] = max(1, min(10, int(sf_m.group(1)))) if sf_m else 0
    raw["cp"] = max(1, min(10, int(cp_m.group(1)))) if cp_m else 0
    raw["rq"] = max(1, min(10, int(rq_m.group(1)))) if rq_m else 0

    final_scaled_score = min(raw["rq"], raw["cp"]) if (raw["rq"] <= 3 or raw["cp"] <= 3) else max(1, min(10, raw["sf"]))

    return {
        "sf": raw["sf"], 
        "cp": raw["cp"], 
        "rq": raw["rq"], 
        "final_scaled_score": final_scaled_score,
        "raw_extraction": raw
    }

def run_inference(model, tokenizer, user_content: list) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = tokenizer(
        text=[text], images=image_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    raw_out = tokenizer.batch_decode(
        out_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

    return raw_out
