"""
Math Problem Solver using Llama-3.1-8B-Instruct
This program predicts answers for math word problems using the Hugging Face transformers library.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional
import re


class MathProblemSolver:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the Math Problem Solver with Llama model.
        
        Args:
            model_name: The Hugging Face model identifier
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully!")
    
    def create_prompt(self, problem: str, options: str) -> str:
        """
        Create a structured prompt for the model.
        
        Args:
            problem: The math problem statement
            options: The multiple choice options
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Solve the following math problem step by step and select the correct answer.

Problem: {problem}

Options: {options}

Please provide:
1. Step-by-step solution
2. Final answer (letter only: a, b, c, d, or e)

Solution:"""
        return prompt
    
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the answer letter from the model's response.
        
        Args:
            response: The model's generated response
            
        Returns:
            The extracted answer letter (a-e) or None
        """
        # Look for patterns like "Answer: a" or "answer is a" or just letter at the end
        patterns = [
            r'[Aa]nswer:\s*([a-eA-E])',
            r'[Aa]nswer\s+is\s+([a-eA-E])',
            r'[Tt]he\s+correct\s+answer\s+is\s+([a-eA-E])',
            r'\b([a-eA-E])\s*$',  # Letter at the end
            r'option\s+([a-eA-E])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        # If no pattern matches, look for the last occurrence of a-e
        letters = re.findall(r'\b([a-eA-E])\b', response)
        if letters:
            return letters[-1].lower()
        
        return None
    
    def solve_problem(self, problem_data: Dict, max_length: int = 512) -> Dict:
        """
        Solve a single math problem.
        
        Args:
            problem_data: Dictionary containing problem, options, and correct answer
            max_length: Maximum length for generated response
            
        Returns:
            Dictionary with prediction and correctness
        """
        problem = problem_data['Problem']
        options = problem_data['options']
        correct_answer = problem_data['correct']
        
        # Create prompt
        prompt = self.create_prompt(problem, options)
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (after the prompt)
        generated_text = response[len(prompt):].strip()
        
        # Extract predicted answer
        predicted_answer = self.extract_answer(generated_text)
        
        return {
            'problem': problem,
            'options': options,
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': predicted_answer == correct_answer if predicted_answer else False,
            'rationale': generated_text,
            'category': problem_data.get('category', 'unknown')
        }
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict:
        """
        Evaluate the model on a dataset of problems.
        
        Args:
            dataset: List of problem dictionaries
            
        Returns:
            Evaluation results with accuracy metrics
        """
        results = []
        correct_count = 0
        total_count = len(dataset)
        
        print(f"\nEvaluating {total_count} problems...")
        
        for idx, problem_data in enumerate(dataset):
            print(f"\nProblem {idx + 1}/{total_count}")
            print(f"Category: {problem_data.get('category', 'unknown')}")
            
            result = self.solve_problem(problem_data)
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            print(f"Correct: {result['correct_answer']} | Predicted: {result['predicted_answer']} | "
                  f"Match: {'✓' if result['is_correct'] else '✗'}")
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Calculate category-wise accuracy
        category_stats = {}
        for result in results:
            cat = result['category']
            if cat not in category_stats:
                category_stats[cat] = {'correct': 0, 'total': 0}
            category_stats[cat]['total'] += 1
            if result['is_correct']:
                category_stats[cat]['correct'] += 1
        
        for cat in category_stats:
            category_stats[cat]['accuracy'] = (
                category_stats[cat]['correct'] / category_stats[cat]['total']
            )
        
        return {
            'results': results,
            'overall_accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'category_stats': category_stats
        }


def main():
    """Main function to run the math problem solver."""
    
    # Load the dataset from uploaded file
    import sys
    
    # Check if dataset file exists
    dataset_path = '/mnt/user-data/uploads'
    
    try:
        # List files in uploads
        import os
        files = os.listdir(dataset_path)
        print(f"Files in uploads: {files}")
        
        # Load the JSON file
        json_file = [f for f in files if f.endswith('.json')][0] if files else None
        
        if json_file:
            with open(os.path.join(dataset_path, json_file), 'r') as f:
                dataset = json.load(f)
            print(f"Loaded {len(dataset)} problems from dataset")
        else:
            print("No JSON file found in uploads directory")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Initialize solver
    solver = MathProblemSolver()
    
    # Evaluate on dataset
    evaluation = solver.evaluate_dataset(dataset)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {evaluation['overall_accuracy']:.2%}")
    print(f"Correct: {evaluation['correct_count']}/{evaluation['total_count']}")
    
    print("\nCategory-wise Accuracy:")
    for cat, stats in evaluation['category_stats'].items():
        print(f"  {cat}: {stats['accuracy']:.2%} "
              f"({stats['correct']}/{stats['total']})")
    
    # Save results
    output_path = '/mnt/user-data/outputs/evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Save detailed report
    report_path = '/mnt/user-data/outputs/detailed_report.txt'
    with open(report_path, 'w') as f:
        f.write("MATH PROBLEM SOLVER - DETAILED REPORT\n")
        f.write("="*80 + "\n\n")
        
        for idx, result in enumerate(evaluation['results']):
            f.write(f"Problem {idx + 1}\n")
            f.write(f"Category: {result['category']}\n")
            f.write(f"Problem: {result['problem']}\n")
            f.write(f"Options: {result['options']}\n")
            f.write(f"Correct Answer: {result['correct_answer']}\n")
            f.write(f"Predicted Answer: {result['predicted_answer']}\n")
            f.write(f"Status: {'CORRECT' if result['is_correct'] else 'INCORRECT'}\n")
            f.write(f"Model Rationale:\n{result['rationale']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()