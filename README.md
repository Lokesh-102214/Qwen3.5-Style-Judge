# Qwen3.5 Style Judge Inference

This repository contains the inference code for the **Qwen3.5-0.8B** visual style judge. The original Kaggle notebook code has been refactored into a modular, easy-to-use local evaluation suite.

## Adapter Models
The fine-tuned LoRA adapters are hosted on HuggingFace:
[GamerQuant/Qwen3.5_0.8b_10kbalanced_k2_lora16](https://huggingface.co/GamerQuant/Qwen3.5_0.8b_10kbalanced_k2_lora16)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Lokesh-102214/Qwen3.5-Style-Judge.git
   cd qwen-style-judge
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Single Set Evaluation
To evaluate a single style transfer (requires 2 exemplar pairs and 1 query pair):

```bash
python infer_single.py \
    --style "acrylic" \
    --ex_a_src path/to/ex1_src.png \
    --ex_a_dst path/to/ex1_dst.png \
    --ex_b_src path/to/ex2_src.png \
    --ex_b_dst path/to/ex2_dst.png \
    --q_src path/to/query_src.png \
    --q_res path/to/query_result.png
```

### Batch Evaluation
To evaluate an entire dataset provided in a CSV file:

```bash
python infer_batch.py \
    --csv_path data/your_dataset.csv \
    --output_path results.csv
```
The CSV should contain the following columns: `style`, `ex_a_src_path`, `ex_a_dst_path`, `ex_b_src_path`, `ex_b_dst_path`, `q_src_path`, `q_res_path`, and optionally `final_score` for computing ground-truth metrics.

## Architecture

- `src/config.py`: Contains system prompts, style marker mappings, and model parameters.
- `src/model.py`: Wraps `unsloth` model and adapter loading.
- `src/prompt.py`: Formats the complex multi-image style marker prompt.
- `src/evaluate.py`: Handles model generation and extracts specific metric integer scores from XML tags.
