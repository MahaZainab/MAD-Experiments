"""
Math Problem Solver using Llama-3.1-8B-Instruct
This program predicts answers for math word problems using the Hugging Face transformers library.
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class MathProblemSolver:
    def __init__(self, model_name: str = "google/flan-t5-xl"):
        """
        Initialize the Math Problem Solver with Llama model.

        Args:
            model_name: The Hugging Face model identifier
        """
        print(f"Loading model: {model_name}")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Some tokenizers (e.g., Llama) don't define pad_token by default.
        if self.tokenizer.pad_token_id is None:
            # Prefer eos_token as pad token when missing.
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Last resort: create a pad token.
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # Choose the correct model class based on architecture.
        cfg = AutoConfig.from_pretrained(model_name)
        self.is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))

        model_kwargs = dict(
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # If we added tokens, resize embeddings.
        if hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Put model on device if device_map wasn't used.
        if not torch.cuda.is_available():
            self.model.to(torch.device("cpu"))

        self.model.eval()
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
            r"[Aa]nswer:\s*([a-eA-E])",
            r"[Aa]nswer\s+is\s+([a-eA-E])",
            r"[Tt]he\s+correct\s+answer\s+is\s+([a-eA-E])",
            r"\b([a-eA-E])\s*$",  # Letter at the end
            r"option\s+([a-eA-E])",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        # If no pattern matches, look for the last occurrence of a-e
        letters = re.findall(r"\b([a-eA-E])\b", response)
        if letters:
            return letters[-1].lower()

        return None

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generate text from the model given a prompt.
        Handles both causal LM and encoder-decoder models.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            if self.is_encoder_decoder:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return text
            else:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return text

    def solve_problem(self, problem_data: Dict, max_length: int = 512) -> Dict:
        """
        Solve a single math problem.

        Args:
            problem_data: Dictionary containing problem, options, and correct answer
            max_length: Maximum length for generated response

        Returns:
            Dictionary with prediction and correctness
        """
        problem = problem_data["Problem"]
        options = problem_data["options"]
        correct_answer = problem_data["correct"]

        # Create prompt
        prompt = self.create_prompt(problem, options)

        # Generate
        response = self._generate(prompt, max_new_tokens=max_length)

        # Extract just the generated part (after the prompt) when applicable
        if (not self.is_encoder_decoder) and response.startswith(prompt):
            generated_text = response[len(prompt) :].strip()
        else:
            generated_text = response.strip()

        # Extract predicted answer
        predicted_answer = self.extract_answer(generated_text)

        return {
            "problem": problem,
            "options": options,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == correct_answer if predicted_answer else False,
            "rationale": generated_text,
            "category": problem_data.get("category", "unknown"),
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

            if result["is_correct"]:
                correct_count += 1

            print(
                f"Correct: {result['correct_answer']} | Predicted: {result['predicted_answer']} | "
                f"Match: {'✓' if result['is_correct'] else '✗'}"
            )

        accuracy = correct_count / total_count if total_count > 0 else 0

        # Calculate category-wise accuracy
        category_stats: Dict[str, Dict[str, float]] = {}
        for result in results:
            cat = result["category"]
            if cat not in category_stats:
                category_stats[cat] = {"correct": 0, "total": 0}
            category_stats[cat]["total"] += 1
            if result["is_correct"]:
                category_stats[cat]["correct"] += 1

        for cat in category_stats:
            category_stats[cat]["accuracy"] = category_stats[cat]["correct"] / category_stats[cat]["total"]

        return {
            "results": results,
            "overall_accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "category_stats": category_stats,
        }


def _find_dataset_json(dataset_path: str) -> str:
    """
    Find a JSON dataset file within dataset_path.

    Priority:
      1) If dataset_path is a file and endswith .json, use it.
      2) If dataset_path is a directory, use the first .json file (sorted).
    """
    if os.path.isfile(dataset_path):
        if dataset_path.lower().endswith(".json"):
            return dataset_path
        raise FileNotFoundError(f"Provided dataset path is a file but not a .json: {dataset_path}")

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(".json")])
    if not files:
        raise FileNotFoundError(f"No .json files found in directory: {dataset_path}")
    return os.path.join(dataset_path, files[0])


def main():
    """Main function to run the math problem solver."""

    # Allow passing dataset path and model name via CLI
    # Usage: python math_problem_solver_fixed.py [dataset_path] [model_name]
    dataset_path = sys.argv[1] if len(sys.argv) >= 2 else "dataset"
    model_name = sys.argv[2] if len(sys.argv) >= 3 else "google/flan-t5-xl"

    try:
        json_path = _find_dataset_json(dataset_path)
        print(f"Loading dataset from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError("Dataset JSON must be a list of problem objects.")

        print(f"Loaded {len(dataset)} problems from dataset")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Initialize solver
    solver = MathProblemSolver(model_name=model_name)

    # Evaluate on dataset
    evaluation = solver.evaluate_dataset(dataset)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nOverall Accuracy: {evaluation['overall_accuracy']:.2%}")
    print(f"Correct: {evaluation['correct_count']}/{evaluation['total_count']}")

    print("\nCategory-wise Accuracy:")
    for cat, stats in evaluation["category_stats"].items():
        print(f"  {cat}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Save results
    output_path = "flan_evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save detailed report
    report_path = "flan_detailed_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MATH PROBLEM SOLVER - DETAILED REPORT\n")
        f.write("=" * 80 + "\n\n")

        for idx, result in enumerate(evaluation["results"]):
            f.write(f"Problem {idx + 1}\n")
            f.write(f"Category: {result['category']}\n")
            f.write(f"Problem: {result['problem']}\n")
            f.write(f"Options: {result['options']}\n")
            f.write(f"Correct Answer: {result['correct_answer']}\n")
            f.write(f"Predicted Answer: {result['predicted_answer']}\n")
            f.write(f"Status: {'CORRECT' if result['is_correct'] else 'INCORRECT'}\n")
            f.write(f"Model Rationale:\n{result['rationale']}\n")
            f.write("\n" + "-" * 80 + "\n\n")

    print(f"Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
