#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys

# Import your existing clahe function
# Ensure src is in your path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.features.extractor import clahe_enhancement

def process_single_image(args):
    img_path, output_path, repeat = args
    try:
        img = Image.open(img_path).convert('RGB')
        enhanced = clahe_enhancement(img, repeat_clahe=repeat)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch apply CLAHE to a dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--repeat-clahe", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply CLAHE twice for stronger effect")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    
    # Gather all image paths
    image_paths = list(input_root.rglob("*.jpg")) + list(input_root.rglob("*.png"))
    
    # Prepare task arguments
    tasks = []
    for p in image_paths:
        relative_path = p.relative_to(input_root)
        out_p = output_root / relative_path
        tasks.append((p, out_p, args.repeat_clahe))

    print(f"Processing {len(tasks)} images using {cpu_count()} cores...")
    
    # Run with multiprocessing
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks)))

    print(f"Preprocessing complete. Images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()