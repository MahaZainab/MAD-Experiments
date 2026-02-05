"""
CruxEval (code+input -> output) Solver using Hugging Face Transformers.

Run:
  python cruxeval_prediction_qwen257b.py

What it does:
- Loads DEFAULT_DATASET (cruxeval.json) which is a list of examples with keys:
    "code", "input", "output", "id" (and sometimes "category")
- Prompts the model with code+input and asks it to return ONLY JSON:
    {"predicted_output": "...", "explanation": "..."}
- Stores predicted output + explanation + raw model text
- Computes strict and normalized (strip-only) accuracy
- Writes:
    DEFAULT_RESULTS_JSON
    DEFAULT_REPORT_TXT

Important dataset note:
- The dataset stores escaped newlines/tabs as literal backslash escapes (e.g. "\\n", "\\t"),
  NOT as actual newline/tab characters. This script DOES NOT convert them.
"""

import json
import re
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Defaults (edit once here)
# -------------------------
DEFAULT_DATASET = "cruxeval.json"              # or "cruxeval_mini.json"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_RESULTS_JSON = "cruxeval_qwen257b_results.json"
DEFAULT_REPORT_TXT = "cruxeval_qwen257b_report.txt"


class CruxEvalSolver:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Ensure generate doesn't crash if pad token is unset
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("Model loaded successfully!")

    def create_prompt(self, code: str, inp: str) -> str:
        """
        Prompt for JSON-only output to keep predicted stdout separate from explanation.

        We explicitly instruct:
        - predicted_output must be EXACT Python repr style (strings with quotes)
        - Preserve literal backslash escapes like \n or \t as two characters.
        """
        return f"""You are given Python code and input arguments.

Task:
Predict the exact return value of the function.

Definition of explanation:
- A short, high-level description of your chain of thought. How did you arrive at the answer?
- Make it concise.
- Use 2-3 sentences to explain your reasoning.

Rules for predicted_output:
- Return the value exactly as Python would display it (repr-style).
- If it is a string, include quotes.
- If it contains backslash escapes like \\n or \\t, return them as literal characters (backslash+n), not as real newlines/tabs.

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "explanation": string
}}

Code:
```python
{code}
```

Input:
{inp}
"""

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """
        Parse model output as JSON. If extra text exists, extract the first {...} block.
        """
        text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        return None

    def extract_predicted_output_and_explanation(self, generated_text: str) -> Tuple[str, str]:
        obj = self._extract_json(generated_text)
        if obj is None:
            # Worst-case fallback: treat everything as output
            return generated_text.strip(), ""

        pred = obj.get("predicted_output", "")
        expl = obj.get("explanation", "")

        if pred is None:
            pred = ""
        if expl is None:
            expl = ""
        return str(pred), str(expl)

    @staticmethod
    def normalize_output(s: str) -> str:
        """
        For this dataset, outputs generally do NOT contain real newlines.
        Keep normalization minimal to avoid changing meaning.
        """
        if s is None:
            return ""
        return s.strip()

    def solve_one(self, example: Dict, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Dict:
        """
        Solve a single CruxEval example.
        Expected keys: code, input, output
        """
        code = example.get("code", "")
        inp = example.get("input", "")
        gold = example.get("output", "")
        ex_id = example.get("id", "")

        prompt = self.create_prompt(code, inp)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,      # deterministic for eval
                temperature=0.0,      # extra safety (ignored by some models when do_sample=False)
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Try to isolate the newly generated text.
        # If the model echoed the prompt, slicing by prompt length works well.
        generated_text = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

        pred_output, explanation = self.extract_predicted_output_and_explanation(generated_text)

        strict_correct = pred_output == gold
        normalized_correct = self.normalize_output(pred_output) == self.normalize_output(gold)

        return {
            "id": ex_id,
            "code": code,
            "input": inp,
            "gold_output": gold,
            "predicted_output": pred_output,
            "explanation": explanation,
            "raw_model_text": generated_text,
            "strict_correct": strict_correct,
            "normalized_correct": normalized_correct,
            "category": example.get("category", "unknown"),
        }

    def evaluate_dataset(self, dataset: List[Dict], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Dict:
        results: List[Dict] = []
        strict_correct = 0
        norm_correct = 0
        total = len(dataset)

        print(f"\nEvaluating {total} examples...")

        for idx, ex in enumerate(dataset):
            ex_id = ex.get("id", f"idx_{idx}")
            print(f"\nExample {idx + 1}/{total} | id: {ex_id}")

            r = self.solve_one(ex, max_new_tokens=max_new_tokens)
            results.append(r)

            if r["strict_correct"]:
                strict_correct += 1
            if r["normalized_correct"]:
                norm_correct += 1

            print(
                f"Strict: {'✓' if r['strict_correct'] else '✗'} | "
                f"Normalized: {'✓' if r['normalized_correct'] else '✗'}"
            )

        strict_acc = strict_correct / total if total else 0.0
        norm_acc = norm_correct / total if total else 0.0

        # Optional category-wise stats
        category_stats = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in category_stats:
                category_stats[cat] = {"strict_correct": 0, "normalized_correct": 0, "total": 0}
            category_stats[cat]["total"] += 1
            if r["strict_correct"]:
                category_stats[cat]["strict_correct"] += 1
            if r["normalized_correct"]:
                category_stats[cat]["normalized_correct"] += 1

        for cat, stats in category_stats.items():
            t = stats["total"]
            stats["strict_accuracy"] = stats["strict_correct"] / t if t else 0.0
            stats["normalized_accuracy"] = stats["normalized_correct"] / t if t else 0.0

        return {
            "results": results,
            "strict_accuracy": strict_acc,
            "normalized_accuracy": norm_acc,
            "strict_correct_count": strict_correct,
            "normalized_correct_count": norm_correct,
            "total_count": total,
            "category_stats": category_stats,
        }


def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of examples.")
    return data


def main():
    dataset_path = DEFAULT_DATASET
    model_name = DEFAULT_MODEL_NAME
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    print(f"Dataset: {dataset_path}")
    print(f"Model:   {model_name}")

    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} examples")

    solver = CruxEvalSolver(model_name=model_name)
    evaluation = solver.evaluate_dataset(dataset, max_new_tokens=max_new_tokens)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nStrict Accuracy: {evaluation['strict_accuracy']:.2%} "
          f"({evaluation['strict_correct_count']}/{evaluation['total_count']})")
    print(f"Normalized Accuracy: {evaluation['normalized_accuracy']:.2%} "
          f"({evaluation['normalized_correct_count']}/{evaluation['total_count']})")

    # Save JSON results
    with open(DEFAULT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {DEFAULT_RESULTS_JSON}")

    # Save detailed text report
    with open(DEFAULT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("CRUXEVAL SOLVER - DETAILED REPORT\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(evaluation["results"], start=1):
            f.write(f"Example {i}\n")
            f.write(f"ID: {r.get('id', '')}\n")
            f.write(f"Category: {r.get('category', 'unknown')}\n")
            f.write("Code:\n")
            f.write(r.get("code", "") + "\n\n")
            f.write(f"Input:\n{r.get('input', '')}\n\n")
            f.write(f"Gold Output:\n{r.get('gold_output', '')}\n\n")
            f.write(f"Predicted Output:\n{r.get('predicted_output', '')}\n\n")
            f.write(f"Strict: {'CORRECT' if r.get('strict_correct') else 'INCORRECT'}\n")
            f.write(f"Normalized: {'CORRECT' if r.get('normalized_correct') else 'INCORRECT'}\n")
            f.write("\nExplanation (model-provided):\n")
            f.write(r.get("explanation", "") + "\n")
            f.write("\nRaw model text:\n")
            f.write(r.get("raw_model_text", "") + "\n")
            f.write("\n" + "-" * 80 + "\n\n")

    print(f"Detailed report saved to {DEFAULT_REPORT_TXT}")


if __name__ == "__main__":
    main()
