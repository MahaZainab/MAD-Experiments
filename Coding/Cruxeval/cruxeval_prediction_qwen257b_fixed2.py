
# Importing libraries
import json
import re
from typing import Dict, List, Tuple, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Dataset and model 
DEFAULT_DATASET = "cruxeval_mini.json"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_RESULTS_JSON = "cruxeval_qwen257b_results.json"
DEFAULT_REPORT_TXT = "cruxeval_qwen257b_report.txt"


class StopAfterFirstJSONObject(StoppingCriteria):
    """
    Stop generation once the newly generated text contains one balanced JSON object.

    Important: We only examine tokens generated AFTER the prompt. This avoids the prompt's
    schema braces from affecting brace counts.
    """
    def __init__(self, tokenizer: AutoTokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode only the newly generated part to avoid braces in the prompt/schema.
        gen_ids = input_ids[0, self.prompt_len:]
        if gen_ids.numel() == 0:
            return False

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Quick exits: if we haven't even started JSON, don't stop.
        if "{" not in text:
            return False

        # Find the first '{' and check if we have a balanced object after it.
        start = text.find("{")
        sub = text[start:]

        depth = 0
        in_str = False
        esc = False

        for ch in sub:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True

        return False


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
        # Prompt kept exactly as in your original script (no semantic changes).
        return f"""You are given Python code and input arguments.

Task:
Your job is to predict the exact output and return that output along with the explanation.

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
    def _first_json_object(text: str) -> Optional[dict]:
        """
        Extract the FIRST valid JSON dict from model output.
        Handles:
        - leading/trailing text
        - ```json fences
        - repeated JSON objects
        """
        if not text:
            return None

        t = text.strip()

        # 1) Direct parse
        try:
            obj = json.loads(t)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # 2) If fenced
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            try:
                obj = json.loads(fence.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # 3) Scan for the first valid dict by trying substrings.
        # Prefer earliest '{' and shortest successful parse.
        starts = [m.start() for m in re.finditer(r"\{", t)]
        for i in starts:
            # try increasing end positions where we see a closing brace
            for j in range(i + 1, len(t)):
                if t[j] != "}":
                    continue
                chunk = t[i : j + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
        return None

    def extract_output_and_explanation(self, generated_text: str) -> Tuple[str, str]:
        obj = self._first_json_object(generated_text)
        if obj is None:
            return "", ""

        out = obj.get("predicted_output", obj.get("output", ""))
        expl = obj.get("explanation", "")

        if out is None:
            out = ""
        if expl is None:
            expl = ""
        return str(out), str(expl)

    @staticmethod
    def normalize_output(s: str) -> str:
        if s is None:
            return ""
        return s.strip()

    def solve_one(self, example: Dict, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Dict:
        code = example.get("code", "")
        inp = example.get("input", "")
        gold = example.get("output", "")
        ex_id = example.get("id", "")

        prompt = self.create_prompt(code, inp)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        stopping = StoppingCriteriaList([StopAfterFirstJSONObject(self.tokenizer, prompt_len)])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )

        # Token-based isolation of generated text
        gen_ids = outputs[0][prompt_len:]
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        pred_output, explanation = self.extract_output_and_explanation(generated_text)

        strict_correct = pred_output == gold
        normalized_correct = self.normalize_output(pred_output) == self.normalize_output(gold)

        return {
            "id": ex_id,
            "code": code,
            "input": inp,
            "gold_output": gold,
            # user-facing fields:
            "output": pred_output,
            "explanation": explanation,
            # backward compat (if you have downstream code expecting it)
            "predicted_output": pred_output,
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

    with open(DEFAULT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {DEFAULT_RESULTS_JSON}")

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
            f.write(f"Output:\n{r.get('output', '')}\n\n")
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
