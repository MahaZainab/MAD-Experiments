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
DEFAULT_JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Can use same or different model for judging
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_JUDGE_MAX_TOKENS = 512
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


class LLMJudge:
    """
    LLM-based judge to evaluate if predicted output matches gold output semantically.
    """
    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL_NAME, use_separate_model: bool = False):
        """
        Args:
            model_name: The model to use for judging
            use_separate_model: If True, loads a separate model instance for judging.
                               If False, will share the model with CruxEvalSolver (more memory efficient)
        """
        self.use_separate_model = use_separate_model
        
        if use_separate_model:
            print(f"Loading separate judge model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print("Judge model loaded successfully!")
        else:
            self.tokenizer = None
            self.model = None
    
    def set_shared_model(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
        """Set shared model and tokenizer from CruxEvalSolver"""
        if not self.use_separate_model:
            self.tokenizer = tokenizer
            self.model = model
    
    def create_judge_prompt(self, code: str, inp: str, gold_output: str, predicted_output: str) -> str:
        """Create prompt for LLM judge to evaluate the prediction"""
        return f"""You are an expert Python code evaluator. Your task is to determine if a predicted output matches the expected output for a given Python code execution.

Code:
```python
{code}
```

Input:
{inp}

Expected Output (Gold Standard):
{gold_output}

Predicted Output:
{predicted_output}

Your task:
Evaluate whether the predicted output is semantically correct compared to the expected output. Consider:
1. Exact matches are correct
2. Semantically equivalent outputs are correct (e.g., different representations of the same value)
3. Different data types representing the same value may be correct (e.g., 1 vs 1.0)
4. Whitespace differences alone should not make outputs incorrect
5. String representation differences that don't change meaning (e.g., quotes) may be acceptable

Return ONLY valid JSON with exactly these keys:
{{
  "is_correct": boolean,
  "confidence": string
}}

Where:
- is_correct: true if predicted output is correct, false otherwise
- confidence: "high", "medium", or "low" based on how certain you are
"""
    
    def judge(self, code: str, inp: str, gold_output: str, predicted_output: str, 
              max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS) -> Dict:
        """
        Use LLM to judge if the predicted output is correct.
        
        Returns:
            Dict with keys: is_correct (bool), confidence (str), raw_judge_text (str)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Judge model not initialized. Call set_shared_model() first.")
        
        prompt = self.create_judge_prompt(code, inp, gold_output, predicted_output)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        
        stopping = StoppingCriteriaList([StopAfterFirstJSONObject(self.tokenizer, prompt_len)])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
        
        gen_ids = outputs[0][prompt_len:]
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        # Parse the judge's response
        judge_result = self._parse_judge_response(generated_text)
        judge_result["raw_judge_text"] = generated_text
        
        return judge_result
    
    @staticmethod
    def _first_json_object(text: str) -> Optional[dict]:
        """Extract the FIRST valid JSON dict from model output."""
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

        # 3) Scan for the first valid dict
        starts = [m.start() for m in re.finditer(r"\{", t)]
        for i in starts:
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
    
    def _parse_judge_response(self, generated_text: str) -> Dict:
        """Parse the judge's JSON response"""
        obj = self._first_json_object(generated_text)
        
        if obj is None:
            # Fallback if JSON parsing fails
            return {
                "is_correct": False,
                "confidence": "low"
            }
        
        is_correct = obj.get("is_correct", False)
        confidence = obj.get("confidence", "medium")
        
        # Ensure correct types
        if not isinstance(is_correct, bool):
            is_correct = str(is_correct).lower() in ["true", "yes", "1"]
        
        return {
            "is_correct": is_correct,
            "confidence": str(confidence)
        }


class CruxEvalSolver:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, 
                 judge_model_name: str = DEFAULT_JUDGE_MODEL_NAME,
                 use_separate_judge: bool = False):
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
        
        # Initialize LLM judge
        self.judge = LLMJudge(model_name=judge_model_name, use_separate_model=use_separate_judge)
        if not use_separate_judge:
            self.judge.set_shared_model(self.tokenizer, self.model)

    def create_prompt(self, code: str, inp: str) -> str:
        # Prompt kept exactly as in your original script (no semantic changes).
        return f"""You are given Python code and input arguments.

Task:
Your job is to predict the exact output and return that output along with the explanation.

Definition of explanation:
- A short, high-level description of your chain of thought. How did you arrive at the answer?
- Make it concise.
- Use 2-3 sentences to explain your reasoning about how you reached to the output.

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

    def solve_one(self, example: Dict, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                  judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS) -> Dict:
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

        # Use LLM judge to evaluate the prediction
        judge_result = self.judge.judge(code, inp, gold, pred_output, max_tokens=judge_max_tokens)

        return {
            "id": ex_id,
            "code": code,
            "input": inp,
            "gold_output": gold,
            "output": pred_output,
            "explanation": explanation,
            "raw_model_text": generated_text,
            # LLM judge results
            "is_correct": judge_result["is_correct"],
            "judge_confidence": judge_result["confidence"],
            "raw_judge_text": judge_result["raw_judge_text"],
            "category": example.get("category", "unknown"),
        }

    def evaluate_dataset(self, dataset: List[Dict], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                        judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS) -> Dict:
        results: List[Dict] = []
        correct_count = 0
        total = len(dataset)

        print(f"\nEvaluating {total} examples...")

        for idx, ex in enumerate(dataset):
            ex_id = ex.get("id", f"idx_{idx}")
            print(f"\nExample {idx + 1}/{total} | id: {ex_id}")

            r = self.solve_one(ex, max_new_tokens=max_new_tokens, judge_max_tokens=judge_max_tokens)
            results.append(r)

            if r["is_correct"]:
                correct_count += 1

            print(
                f"Result: {'✓ CORRECT' if r['is_correct'] else '✗ INCORRECT'} | "
                f"Confidence: {r['judge_confidence']}"
            )

        accuracy = correct_count / total if total else 0.0

        # Category statistics
        category_stats = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in category_stats:
                category_stats[cat] = {
                    "correct": 0, 
                    "total": 0,
                    "high_confidence": 0,
                    "medium_confidence": 0,
                    "low_confidence": 0
                }
            category_stats[cat]["total"] += 1
            if r["is_correct"]:
                category_stats[cat]["correct"] += 1
            
            # Track confidence levels
            conf = r.get("judge_confidence", "medium").lower()
            if "high" in conf:
                category_stats[cat]["high_confidence"] += 1
            elif "low" in conf:
                category_stats[cat]["low_confidence"] += 1
            else:
                category_stats[cat]["medium_confidence"] += 1

        for cat, stats in category_stats.items():
            t = stats["total"]
            stats["accuracy"] = stats["correct"] / t if t else 0.0

        return {
            "results": results,
            "accuracy": accuracy,
            "correct_count": correct_count,
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
    judge_model_name = DEFAULT_JUDGE_MODEL_NAME
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    judge_max_tokens = DEFAULT_JUDGE_MAX_TOKENS
    use_separate_judge = False  # Set to True to use a separate model instance for judging

    print(f"Dataset: {dataset_path}")
    print(f"Model:   {model_name}")
    print(f"Judge Model: {judge_model_name}")
    print(f"Using separate judge model: {use_separate_judge}")

    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} examples")

    solver = CruxEvalSolver(
        model_name=model_name, 
        judge_model_name=judge_model_name,
        use_separate_judge=use_separate_judge
    )
    evaluation = solver.evaluate_dataset(
        dataset, 
        max_new_tokens=max_new_tokens,
        judge_max_tokens=judge_max_tokens
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nAccuracy: {evaluation['accuracy']:.2%} "
          f"({evaluation['correct_count']}/{evaluation['total_count']})")
    
    # Print category statistics
    print("\nCategory Breakdown:")
    print("-" * 80)
    for cat, stats in evaluation['category_stats'].items():
        print(f"{cat}:")
        print(f"  Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        print(f"  Confidence: High={stats['high_confidence']}, "
              f"Medium={stats['medium_confidence']}, Low={stats['low_confidence']}")

    with open(DEFAULT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {DEFAULT_RESULTS_JSON}")

    with open(DEFAULT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("CRUXEVAL SOLVER - DETAILED REPORT (LLM Judge)\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(evaluation["results"], start=1):
            f.write(f"Example {i}\n")
            f.write(f"ID: {r.get('id', '')}\n")
            f.write(f"Category: {r.get('category', 'unknown')}\n")
            f.write("Code:\n")
            f.write(r.get("code", "") + "\n\n")
            f.write(f"Input:\n{r.get('input', '')}\n\n")
            f.write(f"Gold Output:\n{r.get('gold_output', '')}\n\n")
            f.write(f"Predicted Output:\n{r.get('output', '')}\n\n")
            f.write(f"Result: {'CORRECT' if r.get('is_correct') else 'INCORRECT'}\n")
            f.write(f"Judge Confidence: {r.get('judge_confidence', 'unknown')}\n")
            f.write("\nModel Explanation:\n")
            f.write(r.get("explanation", "") + "\n")
            f.write("\nRaw Model Text:\n")
            f.write(r.get("raw_model_text", "") + "\n")
            f.write("\nRaw Judge Text:\n")
            f.write(r.get("raw_judge_text", "") + "\n")
            f.write("\n" + "-" * 80 + "\n\n")

    print(f"Detailed report saved to {DEFAULT_REPORT_TXT}")


if __name__ == "__main__":
    main()