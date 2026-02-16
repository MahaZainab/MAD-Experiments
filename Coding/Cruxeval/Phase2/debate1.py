# =============================================================================
# CruxEval Multi-Model Evaluator  –  with Theory-of-Mind Debate (Phase 2)
#
# Models:  M = Mistral-7B-Instruct-v0.3
#          P = Phi-4-mini-instruct
#          Q = Qwen2.5-7B-Instruct
# Judge:   Qwen2.5-7B-Instruct
#
# Phase 1 – Each agent independently predicts the output.
# Phase 2 – Each agent sees ALL three Phase-1 answers and revises its answer
#            using a Theory-of-Mind debate prompt.
# Phase 3 – Judge evaluates BOTH the Phase-1 and Phase-2 predictions.
# Phase 4 – Results written to JSON + CSV.
#
# Input:  cruxeval_mini.json
# CSV columns (Phase 1 + Phase 2 side-by-side):
#   code, input, gold_output,
#   M_output,  M_reasoning,  M_Correctness,
#   P_output,  P_reasoning,  P_Correctness,
#   Q_output,  Q_reasoning,  Q_Correctness,
#   M_output_d1, M_reasoning_d1, M_Correctness_d1,
#   P_output_d1, P_reasoning_d1, P_Correctness_d1,
#   Q_output_d1, Q_reasoning_d1, Q_Correctness_d1
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
DEFAULT_DATASET        = "cruxeval_mini.json"     # input dataset
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_RESULTS_JSON   = "cruxeval_debate_results.json"
DEFAULT_RESULTS_CSV    = "cruxeval_debate_results.csv"

# Contestant models  (M / P / Q)
MODEL_M = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_P = "microsoft/Phi-4-mini-instruct"
MODEL_Q = "Qwen/Qwen2.5-7B-Instruct"

# Judge model
MODEL_JUDGE = "Qwen/Qwen2.5-7B-Instruct"

# Human-readable labels
CONTESTANT_MODELS = [
    ("M", MODEL_M),
    ("P", MODEL_P),
    ("Q", MODEL_Q),
]

# Labels for the other two agents seen during debate (Theory-of-Mind)
PEER_LABELS = {
    "M": [("P", "Phi-4-mini"),       ("Q", "Qwen2.5-7B")],
    "P": [("M", "Mistral-7B"),       ("Q", "Qwen2.5-7B")],
    "Q": [("M", "Mistral-7B"),       ("P", "Phi-4-mini")],
}


# ---------------------------------------------------------------------------
# Stopping Criteria
# ---------------------------------------------------------------------------
class StopAfterFirstJSONObject(StoppingCriteria):
    """Stop as soon as the newly generated text contains one balanced JSON object."""

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
            if esc:      esc = False;      continue
            if ch == "\\": esc = True;     continue
            if ch == '"':  in_str = not in_str; continue
            if in_str:     continue
            if ch == "{":  depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False


# ---------------------------------------------------------------------------
# JSON parsing helpers
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
                obj = json.loads(t[i: j + 1])
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


def build_debate_prompt(
    code: str,
    inp: str,
    my_label: str,
    my_output: str,
    my_reasoning: str,
    peer_responses: List[Tuple[str, str, str]],   # [(label, output, reasoning), ...]
) -> str:
    """
    Theory-of-Mind debate prompt.

    Each agent receives:
      - The original code + input
      - Its own Phase-1 answer + reasoning
      - The Phase-1 answers + reasoning of the other two agents

    It must return updated predicted_output + explanation as JSON.
    """
    peer_block = ""
    for peer_label, peer_output, peer_reasoning in peer_responses:
        peer_block += f"""
--- Agent {peer_label} ---
Answer: {peer_output}
Reasoning: {peer_reasoning}
"""

    return f"""You are Agent {my_label}, a debater in a multi-agent debate system for Python output prediction.

You will receive:
1. A Python code snippet and its input.
2. Your own previous answer and reasoning (which may be correct or incorrect).
3. The previous answers and reasoning of the other two agents (which may also be correct or incorrect).

Your job:
- Carefully re-read the code and trace through its execution step by step.
- Consider your own reasoning AND the reasoning provided by the other agents.
- Agents can be wrong — do not blindly follow the majority. Use logical analysis.
- If another agent's reasoning reveals a mistake in your answer, update your answer.
- If you still believe your answer is correct, keep it and explain why.

Rules for predicted_output:
- Return the value exactly as Python would display it (repr-style).
- If it is a string, include quotes.
- If it contains backslash escapes like \\n or \\t, return them as literal characters (backslash+n), not as real newlines/tabs.

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "explanation": string
}}

=== Code ===
```python
{code}
```

=== Input ===
{inp}

=== Your Previous Answer (Agent {my_label}) ===
Answer: {my_output}
Reasoning: {my_reasoning}

=== Other Agents' Answers ===
{peer_block}

Now carefully re-analyse the code and return your (possibly revised) answer as valid JSON.
"""


def build_judge_prompt(code: str, inp: str, gold: str, prediction: str) -> str:
    return f"""You are a automated judge for a Python output prediction task.

Your job is to decide whether the predicted output is CORRECT compared to the gold output.

Rules:
- The UNDERLYING VALUE must match — focus on what the code actually returns.
- Quote-style differences are ACCEPTABLE: if gold is "'hello'" and prediction is
  "hello" (or vice versa), treat them as CORRECT — the underlying string value matches.
- Treat semantically equivalent representations as correct
- Only mark INCORRECT when the actual computed value differs, not just its formatting.
- Return ONLY valid JSON with exactly these keys:
{{
  "verdict": "CORRECT" or "INCORRECT",
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
# Model wrapper  (loads once, runs many, then unloads)
# ---------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, model_name: str):
        print(f"  Loading: {model_name}")
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f"  Loaded:  {model_name}")

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs     = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
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

        gen_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def unload(self):
        """Free GPU memory after use."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dataset loader  — reads CSV file
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> List[Dict]:
    """
    Reads a CSV file. Expected columns: code, input, output (gold).
    Falls back to JSON if the path ends with .json.
    """
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Dataset JSON must be a list of examples.")
        return data

    # CSV path
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append({
                "id":     row.get("id", f"idx_{i}"),
                "code":   row.get("code",   ""),
                "input":  row.get("input",  ""),
                "output": row.get("output", row.get("gold_output", "")),
            })
    return rows


# ---------------------------------------------------------------------------
# Phase 1 – contestant evaluation (one model, all examples)
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
    Indexed identically to dataset.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1 – Contestant [{label}]  {model_name}")
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

        results.append({"output": out, "reasoning": expl, "raw_text": raw})
        print(f"→ predicted: {out[:60]!r}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Phase 2 – Theory-of-Mind debate round (one model, all examples)
# ---------------------------------------------------------------------------
def run_debate_round(
    label: str,
    model_name: str,
    dataset: List[Dict],
    phase1_predictions: Dict[str, List[Dict]],
    max_new_tokens: int,
) -> List[Dict]:
    """
    Each agent re-evaluates its Phase-1 answer after seeing all peers' answers.

    Returns a list of per-example dicts (same structure as run_contestant):
      { "output": ..., "reasoning": ..., "raw_text": ... }
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2 DEBATE – Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        code  = ex.get("code",  "")
        inp   = ex.get("input", "")
        ex_id = ex.get("id",    f"idx_{idx}")

        # Own Phase-1 answer
        my_output    = phase1_predictions[label][idx]["output"]
        my_reasoning = phase1_predictions[label][idx]["reasoning"]

        # Peer Phase-1 answers  (the other two agents)
        peer_responses = []
        for peer_label, peer_name in PEER_LABELS[label]:
            peer_output    = phase1_predictions[peer_label][idx]["output"]
            peer_reasoning = phase1_predictions[peer_label][idx]["reasoning"]
            peer_responses.append((peer_label, peer_output, peer_reasoning))

        prompt = build_debate_prompt(
            code, inp,
            label, my_output, my_reasoning,
            peer_responses,
        )

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj  = _first_json_object(raw)
        out  = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        expl = str(obj.get("explanation", "") or "") if obj else ""

        # Fall back to Phase-1 answer if the model returned nothing useful
        if not out.strip():
            out  = my_output
            expl = f"[debate fallback] {my_reasoning}"

        results.append({"output": out, "reasoning": expl, "raw_text": raw})
        print(f"→ revised: {out[:60]!r}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------
def run_judge(
    dataset: List[Dict],
    predictions: Dict[str, List[Dict]],
    max_new_tokens: int = 128,
) -> Dict[str, List[str]]:
    """
    predictions  – { label: [ {output, reasoning, raw_text}, ... ] }

    Returns { label: [ "CORRECT" | "INCORRECT" | "PARSE_ERROR", ... ] }
    """
    print(f"\n{'='*60}")
    print(f"Running Judge: {MODEL_JUDGE}")
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

            # Empty-output guard
            if not pred.strip():
                gold_is_empty = gold.strip() in ("''", '""', '')
                verdict = "CORRECT" if gold_is_empty else "INCORRECT"
                marker  = "(empty pred, gold also empty → CORRECT)" if gold_is_empty \
                          else "(empty pred → INCORRECT)"
                print(f"    [{label}] pred=''  →  {verdict}  {marker}")
                correctness[label].append(verdict)
                continue

            prompt  = build_judge_prompt(code, inp, gold, pred)
            raw     = runner.generate(prompt, max_new_tokens)

            obj     = _first_json_object(raw)
            verdict = (obj.get("verdict", "") or "").upper() if obj else ""
            if verdict not in ("CORRECT", "INCORRECT"):
                verdict = "INCORRECT"   # safe default

            correctness[label].append(verdict)
            print(f"    [{label}] pred={pred[:40]!r}  →  {verdict}")

    runner.unload()
    return correctness


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_outputs(
    dataset: List[Dict],
    phase1_preds: Dict[str, List[Dict]],
    phase1_corr:  Dict[str, List[str]],
    phase2_preds: Dict[str, List[Dict]],
    phase2_corr:  Dict[str, List[str]],
    json_path: str = DEFAULT_RESULTS_JSON,
    csv_path:  str = DEFAULT_RESULTS_CSV,
) -> None:
    """
    Writes both JSON and CSV with Phase-1 and Phase-2 columns.

    CSV columns:
      code, input, gold_output,
      M_output,    M_reasoning,    M_Correctness,
      P_output,    P_reasoning,    P_Correctness,
      Q_output,    Q_reasoning,    Q_Correctness,
      M_output_d1, M_reasoning_d1, M_Correctness_d1,
      P_output_d1, P_reasoning_d1, P_Correctness_d1,
      Q_output_d1, Q_reasoning_d1, Q_Correctness_d1
    """
    labels = ["M", "P", "Q"]
    rows   = []

    for idx, ex in enumerate(dataset):
        row = {
            "code":        ex.get("code",   ""),
            "input":       ex.get("input",  ""),
            "gold_output": ex.get("output", ""),
        }

        # Phase 1
        for lbl in labels:
            p1 = phase1_preds.get(lbl, [])
            c1 = phase1_corr.get(lbl, [])
            row[f"{lbl}_output"]      = p1[idx]["output"]      if idx < len(p1) else ""
            row[f"{lbl}_reasoning"]   = p1[idx]["reasoning"]   if idx < len(p1) else ""
            row[f"{lbl}_Correctness"] = c1[idx]                if idx < len(c1) else ""

        # Phase 2 (debate round 1)
        for lbl in labels:
            p2 = phase2_preds.get(lbl, [])
            c2 = phase2_corr.get(lbl, [])
            row[f"{lbl}_output_d1"]      = p2[idx]["output"]    if idx < len(p2) else ""
            row[f"{lbl}_reasoning_d1"]   = p2[idx]["reasoning"] if idx < len(p2) else ""
            row[f"{lbl}_Correctness_d1"] = c2[idx]              if idx < len(c2) else ""

        rows.append(row)

    # ---- JSON ---------------------------------------------------------------
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved → {json_path}")

    # ---- CSV ----------------------------------------------------------------
    fieldnames = [
        "code", "input", "gold_output",
        "M_output",    "M_reasoning",    "M_Correctness",
        "P_output",    "P_reasoning",    "P_Correctness",
        "Q_output",    "Q_reasoning",    "Q_Correctness",
        "M_output_d1", "M_reasoning_d1", "M_Correctness_d1",
        "P_output_d1", "P_reasoning_d1", "P_Correctness_d1",
        "Q_output_d1", "Q_reasoning_d1", "Q_Correctness_d1",
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
    phase1_corr: Dict[str, List[str]],
    phase2_corr: Dict[str, List[str]],
    labels: List[str] = ("M", "P", "Q"),
) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Agent':<8} {'Phase-1':>12}   {'Phase-2 (debate)':>18}")
    print("-" * 60)
    for lbl in labels:
        v1 = phase1_corr.get(lbl, [])
        v2 = phase2_corr.get(lbl, [])

        tot1 = len(v1); cor1 = sum(1 for v in v1 if v == "CORRECT")
        tot2 = len(v2); cor2 = sum(1 for v in v2 if v == "CORRECT")

        pct1 = cor1 / tot1 * 100 if tot1 else 0
        pct2 = cor2 / tot2 * 100 if tot2 else 0

        delta = f"{'↑' if pct2 > pct1 else ('↓' if pct2 < pct1 else '→')} {abs(pct2-pct1):.1f}%"
        print(f"  [{lbl}]   {cor1}/{tot1} ({pct1:.1f}%)   {cor2}/{tot2} ({pct2:.1f}%)   {delta}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dataset_path   = DEFAULT_DATASET
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    print(f"Dataset : {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Loaded  : {len(dataset)} examples\n")

    # -----------------------------------------------------------------------
    # PHASE 1 – Each agent independently predicts the output
    # -----------------------------------------------------------------------
    phase1_predictions: Dict[str, List[Dict]] = {}
    for label, model_name in CONTESTANT_MODELS:
        phase1_predictions[label] = run_contestant(
            label, model_name, dataset, max_new_tokens
        )

    # -----------------------------------------------------------------------
    # PHASE 1 JUDGING – Judge evaluates Phase-1 predictions
    # -----------------------------------------------------------------------
    phase1_correctness = run_judge(dataset, phase1_predictions, max_new_tokens=128)

    # -----------------------------------------------------------------------
    # PHASE 2 – Theory-of-Mind debate: each agent sees all Phase-1 answers
    # -----------------------------------------------------------------------
    phase2_predictions: Dict[str, List[Dict]] = {}
    for label, model_name in CONTESTANT_MODELS:
        phase2_predictions[label] = run_debate_round(
            label, model_name, dataset, phase1_predictions, max_new_tokens
        )

    # -----------------------------------------------------------------------
    # PHASE 2 JUDGING – Judge evaluates debate-revised predictions
    # -----------------------------------------------------------------------
    phase2_correctness = run_judge(dataset, phase2_predictions, max_new_tokens=128)

    # -----------------------------------------------------------------------
    # PHASE 4 – Write outputs (JSON + CSV)
    # -----------------------------------------------------------------------
    write_outputs(
        dataset,
        phase1_predictions, phase1_correctness,
        phase2_predictions, phase2_correctness,
        json_path=DEFAULT_RESULTS_JSON,
        csv_path=DEFAULT_RESULTS_CSV,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_summary(phase1_correctness, phase2_correctness)


if __name__ == "__main__":
    main()