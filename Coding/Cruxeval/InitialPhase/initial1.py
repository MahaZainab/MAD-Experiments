# =============================================================================
# CruxEval Multi-Model Evaluator  (FIXED)
# Models:  M = Mistral-7B-Instruct-v0.3
#          P = Phi-4 Mini Instruct
#          Q = Qwen3-8B   ← thinking model, needs special handling
# Judge:   Qwen2.5-7B-Instruct
# Outputs: cruxeval_multi_results.json  +  cruxeval_multi_results.csv
#
# ── BUGS FIXED ───────────────────────────────────────────────────────────────
#
# BUG 1 ▸ Qwen3-8B produces empty outputs (most impactful)
#   ROOT CAUSE: Qwen3-8B is a "thinking" model.  It emits a <think>…</think>
#   reasoning block first, then the actual JSON answer.  The model's EOS token
#   (or the stopping criteria) fired at the end of </think>, so generate()
#   returned ONLY the thinking block — the JSON answer was never produced.
#   FIX: For Qwen3, we pass `enable_thinking=False` via
#   `model.generate(..., generation_config=...)` which suppresses the think
#   block entirely.  As a belt-and-suspenders fallback we also strip any
#   leftover <think>…</think> text before parsing.  If the think block IS
#   present in the output we attempt to extract JSON from the part after it.
#
# BUG 2 ▸ Judge marks empty predictions as CORRECT (inflated accuracy)
#   ROOT CAUSE: When the judge received an empty "Predicted Output" it had
#   nothing to compare and defaulted to CORRECT in many cases (28 out of 49
#   CORRECT verdicts in the sample run were for empty predictions).
#   FIX: Hard-code a pre-check in run_judge() — if prediction is empty string
#   the verdict is immediately set to INCORRECT without calling the model.
#
# BUG 3 ▸ Judge too lenient about quote-style differences
#   ROOT CAUSE: The judge prompt said to treat "equivalent representations" as
#   correct.  It treated 'bcksrutq' (no outer quotes, raw string value) as
#   equivalent to "'bcksrutq'" (Python repr with quotes).  These are actually
#   different outputs: one is the string value, the other is the Python repr.
#   FIX: Tighten the judge prompt to explicitly state that Python repr format
#   must be preserved — missing or extra surrounding quotes are INCORRECT.
#
# =============================================================================

import csv
import json
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET        = "cruxeval_mini.json"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_RESULTS_JSON   = "cruxeval_multi_results.json"
DEFAULT_RESULTS_CSV    = "cruxeval_multi_results.csv"

# Contestant models  (M / P / Q)
MODEL_M = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_P = "microsoft/Phi-4-mini-instruct"
MODEL_Q = "Qwen/Qwen3-8B"          # thinking model — needs special handling

# Judge model (unchanged from original)
MODEL_JUDGE = "Qwen/Qwen2.5-7B-Instruct"

# Set of model names that use Qwen3-style thinking tokens
THINKING_MODELS = {MODEL_Q}

# Human-readable labels used throughout the code
CONTESTANT_MODELS = [
    ("M", MODEL_M),
    ("P", MODEL_P),
    ("Q", MODEL_Q),
]


# ---------------------------------------------------------------------------
# Stopping Criteria – stop as soon as the first balanced JSON object is done
# ---------------------------------------------------------------------------
class StopAfterFirstJSONObject(StoppingCriteria):
    """
    Stop generation once the newly generated text (tokens AFTER the prompt)
    contains one balanced JSON object.
    """

    def __init__(self, tokenizer: AutoTokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer  = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.prompt_len:]
        if gen_ids.numel() == 0:
            return False

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "{" not in text:
            return False

        start = text.find("{")
        sub   = text[start:]
        depth, in_str, esc = 0, False, False

        for ch in sub:
            if esc:          esc = False; continue
            if ch == "\\":   esc = True;  continue
            if ch == '"':    in_str = not in_str; continue
            if in_str:       continue
            if ch == "{":    depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False


# ---------------------------------------------------------------------------
# BUG 1 FIX – strip Qwen3 <think>…</think> from decoded output
# ---------------------------------------------------------------------------
def strip_thinking_block(text: str) -> str:
    """
    Qwen3 sometimes includes a <think>…</think> reasoning block even when
    enable_thinking=False is set (e.g. partial blocks, or when the config
    flag is not fully supported by an older transformers version).
    This strips everything inside <think>…</think> and returns the remainder.
    If no think block is present the text is returned unchanged.
    """
    # Full block: <think>...</think>  (may span multiple lines)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Partial opening tag with no closing (generation was cut inside think)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# JSON parsing helpers  (shared by contestant + judge)
# ---------------------------------------------------------------------------
def _first_json_object(text: str) -> Optional[dict]:
    """Extract the first valid JSON dict from raw model output."""
    if not text:
        return None
    t = text.strip()

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    starts = [m.start() for m in re.finditer(r"\{", t)]
    for i in starts:
        for j in range(i + 1, len(t)):
            if t[j] != "}":
                continue
            try:
                obj = json.loads(t[i : j + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def normalize_output(s: str) -> str:
    return (s or "").strip()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_contestant_prompt(code: str, inp: str) -> str:
    return f"""You are given Python code and input arguments.

Task:
Your job is to predict the exact output and return that output along with the explanation.

Definition of explanation:
- A short, high-level description of your chain of thought. How did you arrive at the answer?
- Make it concise.
- Use 2-3 sentences to explain your reasoning about how you reached to the output.

Rules for predicted_output:
- Return the value exactly as Python would display it (repr-style).
- If it is a string, include the surrounding single quotes exactly as Python repr shows them.
- If it contains backslash escapes like \\n or \\t, return them as literal characters (backslash+n), not as real newlines/tabs.

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": "<value exactly as Python repr>",
  "explanation": "<your reasoning>"
}}

Code:
```python
{code}
```

Input:
{inp}
"""


# BUG 3 FIX: tightened judge prompt — missing/extra quotes are INCORRECT
def build_judge_prompt(code: str, inp: str, gold: str, prediction: str) -> str:
    return f"""You are a strict automated judge for a Python output prediction task.

Your job is to decide whether the predicted output is CORRECT compared to the gold output.

Rules:
- The predicted output must match the gold output in Python repr format exactly.
- Surrounding quotes ARE significant: if gold is "'hello'" (a Python string repr),
  then "hello" (without quotes) is INCORRECT because the format is wrong.
- Ignore only trivial whitespace differences (leading/trailing spaces on the whole value).
- Do NOT treat missing or extra quote characters as equivalent.
- Return ONLY valid JSON with exactly these keys:
{{
  "verdict": "CORRECT" or "INCORRECT",
  "reason": "<one concise sentence>"
}}

Code:
```python
{code}
```

Input:
{inp}

Gold Output:
{gold}

Predicted Output:
{prediction}
"""


# ---------------------------------------------------------------------------
# Model wrapper  (loads once, runs many)
# ---------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, model_name: str):
        print(f"  Loading: {model_name}")
        self.model_name   = model_name
        self.is_thinking  = model_name in THINKING_MODELS
        self.tokenizer    = AutoTokenizer.from_pretrained(model_name)
        self.model        = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f"  Loaded:  {model_name}  (thinking_mode={'ON→disabled' if self.is_thinking else 'N/A'})")

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        # ── BUG 1 FIX: apply chat template for thinking models ──────────────
        # Qwen3 needs the prompt wrapped in its chat template AND
        # enable_thinking=False so it skips the <think> block and goes
        # straight to the answer.
        if self.is_thinking and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,   # disable CoT reasoning block
                )
                inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
            except TypeError:
                # Older transformers may not support enable_thinking kwarg
                # Fall back to standard template without the flag
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        prompt_len = inputs["input_ids"].shape[-1]
        stopping   = StoppingCriteriaList(
            [StopAfterFirstJSONObject(self.tokenizer, prompt_len)]
        )

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

        gen_ids        = outputs[0][prompt_len:]
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # ── BUG 1 FIX (belt & suspenders): strip any residual think block ───
        if self.is_thinking:
            generated_text = strip_thinking_block(generated_text)

        return generated_text

    def unload(self):
        """Free GPU memory after use."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Contestant evaluation  (one model, all examples)
# ---------------------------------------------------------------------------
def run_contestant(
    label: str,
    model_name: str,
    dataset: List[Dict],
    max_new_tokens: int,
) -> List[Dict]:
    """
    Returns a list of per-example dicts:
      { "output": ..., "reasoning": ..., "raw_text": ... }
    indexed identically to dataset.
    """
    print(f"\n{'='*60}")
    print(f"Running contestant  [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        code   = ex.get("code",  "")
        inp    = ex.get("input", "")
        ex_id  = ex.get("id",    f"idx_{idx}")
        prompt = build_contestant_prompt(code, inp)

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj  = _first_json_object(raw)
        out  = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        expl = str(obj.get("explanation", "") or "") if obj else ""

        # Warn loudly if output is still empty (helps catch model-specific issues)
        if not out:
            print(f"\n  ⚠  [{label}] empty output for example {idx+1}. raw={raw[:120]!r}")

        results.append({"output": out, "reasoning": expl, "raw_text": raw})
        print(f"→ predicted: {out[:60]!r}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Judge model  (separate function)
# ---------------------------------------------------------------------------
def run_judge(
    dataset: List[Dict],
    predictions: Dict[str, List[Dict]],
    max_new_tokens: int = 128,
) -> Dict[str, List[str]]:
    """
    predictions  – { label: [ {output, reasoning, raw_text}, ... ] }

    Returns { label: [ "CORRECT" | "INCORRECT", ... ] }

    BUG 2 FIX: Empty predictions are immediately marked INCORRECT without
    sending them to the judge model — this prevents false CORRECT verdicts
    when a model produced no output.
    """
    print(f"\n{'='*60}")
    print(f"Running Judge Model: {MODEL_JUDGE}")
    print(f"{'='*60}")

    runner      = ModelRunner(MODEL_JUDGE)
    correctness: Dict[str, List[str]] = {lbl: [] for lbl in predictions}

    for idx, ex in enumerate(dataset):
        code  = ex.get("code",   "")
        inp   = ex.get("input",  "")
        gold  = ex.get("output", "")
        ex_id = ex.get("id",     f"idx_{idx}")

        print(f"  [Judge] Example {idx+1}/{len(dataset)}  id={ex_id}")

        for label in predictions:
            pred = predictions[label][idx]["output"]

            # ── BUG 2 FIX: never call the judge for empty predictions ───────
            if not pred.strip():
                verdict = "INCORRECT"
                print(f"    [{label}] pred='' → INCORRECT (skipped judge: empty output)")
                correctness[label].append(verdict)
                continue

            prompt  = build_judge_prompt(code, inp, gold, pred)
            raw     = runner.generate(prompt, max_new_tokens)
            obj     = _first_json_object(raw)
            verdict = (obj.get("verdict", "") or "").upper() if obj else ""
            if verdict not in ("CORRECT", "INCORRECT"):
                verdict = "INCORRECT"   # safe default if judge parse fails

            correctness[label].append(verdict)
            print(f"    [{label}] pred={pred[:40]!r}  →  {verdict}")

    runner.unload()
    return correctness


# ---------------------------------------------------------------------------
# Output writers  (separate function)
# ---------------------------------------------------------------------------
def write_outputs(
    dataset: List[Dict],
    predictions: Dict[str, List[Dict]],
    correctness: Dict[str, List[str]],
    json_path: str = DEFAULT_RESULTS_JSON,
    csv_path: str  = DEFAULT_RESULTS_CSV,
) -> None:
    """
    Builds the final row list and writes both JSON and CSV.

    Columns:
      code, input, gold_output,
      M_output, M_reasoning, M_Correctness,
      P_output, P_reasoning, P_Correctness,
      Q_output, Q_reasoning, Q_Correctness
    """
    labels = ["M", "P", "Q"]
    rows   = []

    for idx, ex in enumerate(dataset):
        row = {
            "code":        ex.get("code",   ""),
            "input":       ex.get("input",  ""),
            "gold_output": ex.get("output", ""),
        }
        for lbl in labels:
            preds = predictions.get(lbl, [])
            corrs = correctness.get(lbl, [])
            row[f"{lbl}_output"]      = preds[idx]["output"]    if idx < len(preds) else ""
            row[f"{lbl}_reasoning"]   = preds[idx]["reasoning"] if idx < len(preds) else ""
            row[f"{lbl}_Correctness"] = corrs[idx]              if idx < len(corrs) else ""
        rows.append(row)

    # ── JSON ─────────────────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved → {json_path}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    fieldnames = [
        "code", "input", "gold_output",
        "M_output", "M_reasoning", "M_Correctness",
        "P_output", "P_reasoning", "P_Correctness",
        "Q_output", "Q_reasoning", "Q_Correctness",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV results saved  → {csv_path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(
    correctness: Dict[str, List[str]],
    labels: List[str] = ("M", "P", "Q"),
) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for lbl in labels:
        verdicts = correctness.get(lbl, [])
        total    = len(verdicts)
        correct  = sum(1 for v in verdicts if v == "CORRECT")
        pct      = correct / total * 100 if total else 0
        print(f"  [{lbl}]  {correct}/{total}  ({pct:.1f}%)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of examples.")
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dataset_path   = DEFAULT_DATASET
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    print(f"Dataset : {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Loaded  : {len(dataset)} examples\n")

    # ── Phase 1: run each contestant model ───────────────────────────────────
    predictions: Dict[str, List[Dict]] = {}
    for label, model_name in CONTESTANT_MODELS:
        predictions[label] = run_contestant(label, model_name, dataset, max_new_tokens)

    # ── Phase 2: judge model decides correctness ─────────────────────────────
    correctness = run_judge(dataset, predictions, max_new_tokens=128)

    # ── Phase 3: write outputs (JSON + CSV) ──────────────────────────────────
    write_outputs(
        dataset,
        predictions,
        correctness,
        json_path=DEFAULT_RESULTS_JSON,
        csv_path=DEFAULT_RESULTS_CSV,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(correctness)


if __name__ == "__main__":
    main()