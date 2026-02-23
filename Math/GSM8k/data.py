#!/usr/bin/env python3

"""
Download the GSM8K dataset from Hugging Face
and save it locally in JSON format.
"""

from datasets import load_dataset
import json
import os


def main():
    print("Downloading GSM8K dataset...")
    
    # Load dataset
    dataset = load_dataset("openai/gsm8k", "main")

    output_dir = "gsm8k_json"
    os.makedirs(output_dir, exist_ok=True)

    # Save each split as JSON
    for split in dataset.keys():
        output_path = os.path.join(output_dir, f"gsm8k_{split}.json")
        
        print(f"Saving {split} split to {output_path}...")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset[split].to_list(), f, indent=2, ensure_ascii=False)

    print("Download complete!")
    print(f"Files saved inside: {output_dir}/")


if __name__ == "__main__":
    main()