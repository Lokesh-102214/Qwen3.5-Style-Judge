from PIL import Image
from typing import List, Dict

def build_user_content_3_pairs(
    ex_a_src: Image.Image, ex_a_dst: Image.Image,
    ex_b_src: Image.Image, ex_b_dst: Image.Image,
    ex_c_src: Image.Image, ex_c_dst: Image.Image,
    q_src:    Image.Image, q_res:    Image.Image,
    markers:  str,
) -> List[Dict]:
    
    text = (
        f"STYLE MARKERS (Target Visual Language Attributes):\n"
        f"  • {markers}\n\n"
        f"--- BASELINE (TARGET STYLE DNA) ---\n"
        f"[Role: Exemplar Pair A - Perfect 10/10]: <image_1> (Source) → <image_2> (Target Style)\n"
        f"[Role: Exemplar Pair B - Perfect 10/10]: <image_3> (Source) → <image_4> (Target Style)\n"
        f"[Role: Exemplar Pair C - Perfect 10/10]: <image_5> (Source) → <image_6> (Target Style)\n\n"
        f"--- EVALUATION TARGET ---\n"
        f"[Role: Query to Evaluate]: <image_7> (Query Input) → <image_8> (Query Result)\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Blindly extract the style DNA ONLY from Exemplars A, B, and C (Images 2, 4, and 6).\n"
        f"2. Evaluate the Query Result (Image 8) against that specific baseline.\n"
        f"3. Generate detailed <Thinking_Process> and <Final_Output> reasoning that "
        f"analytically justifies your scores based on specific visual evidence. "
        f"Do NOT name or categorize the artistic style."
    )

    return [
        {"type": "image", "image": ex_a_src},
        {"type": "image", "image": ex_a_dst},
        {"type": "image", "image": ex_b_src},
        {"type": "image", "image": ex_b_dst},
        {"type": "image", "image": ex_c_src},
        {"type": "image", "image": ex_c_dst},
        {"type": "image", "image": q_src},
        {"type": "image", "image": q_res},
        {"type": "text",  "text":  text},
    ]

def build_user_content_2_pairs(
    ex_a_src: Image.Image, ex_a_dst: Image.Image,
    ex_b_src: Image.Image, ex_b_dst: Image.Image,
    q_src:    Image.Image, q_res:    Image.Image,
    markers:  str,
) -> List[Dict]:
    
    text = (
        f"STYLE MARKERS (Target Visual Language Attributes):\n"
        f"  • {markers}\n\n"
        f"--- BASELINE (TARGET STYLE DNA) ---\n"
        f"[Role: Exemplar Pair A - Perfect 10/10]: <image_1> (Source) → <image_2> (Target Style)\n"
        f"[Role: Exemplar Pair B - Perfect 10/10]: <image_3> (Source) → <image_4> (Target Style)\n\n"
        f"--- EVALUATION TARGET ---\n"
        f"[Role: Query to Evaluate]: <image_5> (Query Input) → <image_6> (Query Result)\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Blindly extract the style DNA ONLY from Exemplars A and B (Images 2 and 4).\n"
        f"2. Evaluate the Query Result (Image 6) against that specific baseline.\n"
        f"3. Generate detailed <Thinking_Process> and <Final_Output> reasoning that "
        f"analytically justifies your scores based on specific visual evidence. "
        f"Do NOT name or categorize the artistic style."
    )

    return [
        {"type": "image", "image": ex_a_src},
        {"type": "image", "image": ex_a_dst},
        {"type": "image", "image": ex_b_src},
        {"type": "image", "image": ex_b_dst},
        {"type": "image", "image": q_src},
        {"type": "image", "image": q_res},
        {"type": "text",  "text":  text},
    ]
