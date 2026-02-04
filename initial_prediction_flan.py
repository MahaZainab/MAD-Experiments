import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from tqdm import tqdm

# Set HuggingFace cache directory
os.environ["HF_HOME"] = "/aiau010_scratch/maz0032/.cache/huggingface"

def load_model_and_tokenizer(model_name):
    """
    Load the model and tokenizer from HuggingFace.
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Model loaded successfully on: {model.device}")
    return model, tokenizer

def create_prompt(problem, options):
    """
    Create a prompt for the model with the problem and options.
    """
    prompt = f"""Answer the following multiple-choice math problem. Provide only the letter of the correct answer (a, b, c, d, or e).

Problem: {problem}

Options: {options}

Answer (only the letter):"""
    
    return prompt

def get_model_prediction(model, tokenizer, prompt, max_new_tokens=10):
    """
    Get prediction from the model.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.lower().strip()

    for char in answer:
        if char in ['a', 'b', 'c', 'd', 'e']:
            return char
    return answer[:1] if answer else ""

def main():
    model_name = "google/flan-t5-xl"
    input_file = "train.json"
    output_file = "predictions.csv"
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} problems")
    
    model, tokenizer = load_model_and_tokenizer(model_name)
    results = []
    
    print("\nProcessing problems...")
    for idx, item in enumerate(tqdm(data)):
        problem = item['Problem']
        options = item['options']
        correct_answer = item['correct']
        
        prompt = create_prompt(problem, options)
        prediction = get_model_prediction(model, tokenizer, prompt)
        
        results.append({
            'problem_id': idx,
            'problem': problem,
            'options': options,
            'correct_answer': correct_answer,
            'predicted_answer': prediction,
            'is_correct': prediction == correct_answer,
            'category': item.get('category', 'unknown')
        })
        
        if (idx + 1) % 10 == 0:
            accuracy = sum(1 for r in results if r['is_correct']) / len(results) * 100
            print(f"\nProgress: {idx + 1}/{len(data)} | Current Accuracy: {accuracy:.2f}%")
    
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Total problems: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
