import os
import pandas as pd
from PIL import Image
from .config import IMAGE_SIZE

def load_image(path_or_url: str) -> Image.Image:
    """Helper to load and format an image from a local path or URL."""
    if path_or_url.startswith("http"):
        import requests
        from io import BytesIO
        response = requests.get(path_or_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(path_or_url)
    
    return img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

def load_dataset(csv_path: str, verify_paths: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    if verify_paths:
        def paths_are_valid(row):
            paths = [
                row.get("ex_a_src_path"), row.get("ex_a_dst_path"), 
                row.get("ex_b_src_path"), row.get("ex_b_dst_path"), 
                row.get("q_src_path"),    row.get("q_res_path")
            ]
            # Check if all paths are non-null and either a URL or valid local file
            return all(isinstance(p, str) and (p.startswith("http") or os.path.exists(p)) for p in paths)
        
        df = df[df.apply(paths_are_valid, axis=1)].reset_index(drop=True)
    
    return df
