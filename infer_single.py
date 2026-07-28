import argparse
from src.config import STYLE_DESCRIPTIONS, DEFAULT_BASE_MODEL, DEFAULT_ADAPTER
from src.data import load_image
from src.prompt import build_user_content_2_pairs
from src.model import load_vision_model
from src.evaluate import run_inference, extract_multi_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a single style transfer.")
    parser.add_argument("--style", type=str, required=True, help="Target style (e.g., 'acrylic')")
    parser.add_argument("--ex_a_src", type=str, required=True, help="Exemplar A Source")
    parser.add_argument("--ex_a_dst", type=str, required=True, help="Exemplar A Target")
    parser.add_argument("--ex_b_src", type=str, required=True, help="Exemplar B Source")
    parser.add_argument("--ex_b_dst", type=str, required=True, help="Exemplar B Target")
    parser.add_argument("--q_src", type=str, required=True, help="Query Source")
    parser.add_argument("--q_res", type=str, required=True, help="Query Result to Evaluate")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER)
    
    args = parser.parse_args()
    
    style_desc = STYLE_DESCRIPTIONS.get(args.style, "distinctive textures, colors, and brushstrokes.")
    
    print("Loading images...")
    ex_a_src = load_image(args.ex_a_src)
    ex_a_dst = load_image(args.ex_a_dst)
    ex_b_src = load_image(args.ex_b_src)
    ex_b_dst = load_image(args.ex_b_dst)
    q_src = load_image(args.q_src)
    q_res = load_image(args.q_res)
    
    user_content = build_user_content_2_pairs(
        ex_a_src, ex_a_dst, ex_b_src, ex_b_dst, q_src, q_res, style_desc
    )
    
    model, tokenizer = load_vision_model(args.base_model, args.adapter)
    
    print("\nRunning evaluation...")
    raw_output = run_inference(model, tokenizer, user_content)
    metrics = extract_multi_metrics(raw_output)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Style Fidelity       : {metrics['sf']}/10")
    print(f"Content Preservation : {metrics['cp']}/10")
    print(f"Rendering Quality    : {metrics['rq']}/10")
    print(f"Final Scaled Score   : {metrics['final_scaled_score']}/10")
    print("\n[Raw Model Output]")
    print(raw_output)

if __name__ == "__main__":
    main()
