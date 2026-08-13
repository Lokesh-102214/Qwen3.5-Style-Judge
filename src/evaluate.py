import re
import time
import torch
from threading import Thread, Event
from typing import Dict
from transformers import LogitsProcessorList, LogitsProcessor
from qwen_vl_utils import process_vision_info
from .config import SYSTEM_PROMPT, MAX_NEW_TOKENS

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class GPUPerfMonitor:
    def __init__(self):
        self.stop_event = Event()
        self.utils = []
        self.powers = []
        self.thread = None
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass
    
    def _monitor(self):
        while not self.stop_event.is_set():
            if PYNVML_AVAILABLE and hasattr(self, 'handle'):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 # in Watts
                    self.utils.append(util)
                    self.powers.append(power)
                except Exception:
                    pass
            time.sleep(0.5)

    def start(self):
        if PYNVML_AVAILABLE and hasattr(self, 'handle'):
            self.stop_event.clear()
            self.utils = []
            self.powers = []
            self.thread = Thread(target=self._monitor)
            self.thread.start()

    def stop(self):
        avg_util, avg_power = 0, 0
        if PYNVML_AVAILABLE and self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            if self.utils:
                avg_util = sum(self.utils) / len(self.utils)
            if self.powers:
                avg_power = sum(self.powers) / len(self.powers)
        return avg_util, avg_power

def get_hardware_info():
    info = {
        "pytorch_version": torch.__version__,
        "inference_engine": "transformers (unsloth)",
        "gpu_name": "Unknown"
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info

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

def run_inference(model, tokenizer, user_content: list, return_perf: bool = False):
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

    gpu_monitor = GPUPerfMonitor()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    gpu_monitor.start()

    start_time = time.time()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    total_time = time.time() - start_time
        
    avg_gpu_util, avg_power = gpu_monitor.stop()

    generated_ids = out_ids[:, inputs.input_ids.shape[1]:]
    num_tokens = generated_ids.shape[1] if generated_ids.shape[1] > 0 else 1

    raw_out = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    if not return_perf:
        return raw_out

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) # GB
    else:
        peak_vram = 0

    # Approximate timing metrics since we removed the LogitsProcessor for Unsloth compatibility
    tpot = total_time / num_tokens
    
    perf_metrics = {
        "ttft_sec": 0.0, # Removed to preserve Unsloth speed
        "tpot_sec": tpot,
        "prefill_time_sec": 0.0, # Removed to preserve Unsloth speed
        "total_time_sec": total_time,
        "peak_vram_gb": peak_vram,
        "avg_gpu_utilization_pct": avg_gpu_util,
        "avg_power_consumption_w": avg_power,
        "hardware_info": get_hardware_info()
    }

    return raw_out, perf_metrics
