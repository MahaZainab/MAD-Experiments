# =============================================================================
# CodeQA Multi-Model Evaluator  -  Peer Debate with Confidence Scores
#
# Dataset : CodeQA  (each example has: code, question, answer)
#           The task is to answer a natural-language question about a code
#           snippet (e.g. "What does the code do?", "Does it raise an error?")
#
# Models:
#   M     = Mistral-7B-Instruct-v0.3         (student agent)
#   P     = Phi-4-mini-instruct               (student agent)
#   Q     = Qwen2.5-7B-Instruct               (student agent)
#   Judge = Qwen2.5-7B-Instruct               (evaluator, unchanged)
#
# Pipeline
# --------
# Phase 0      : Each agent independently answers the question AND provides
#                a confidence score (0.0 – 1.0).
#                --> Judge scores Phase 0
#
# Rounds 1-4   : Pure peer-debate (no teacher at any stage)
#
#     Step A  : Each agent sees its own previous answer + confidence score,
#               plus both peers' previous answers + confidence scores,
#               then revises its answer and emits an updated confidence score.
#     Step B  : Judge scores revised answers.
#
# Output    : Single JSON + CSV with ALL rounds side-by-side.
#
# CSV columns
# -----------
# code, question, gold_answer,
# M_output,    M_confidence,    M_reasoning,    M_Correctness,    <- Phase 0
# P_output,    P_confidence,    P_reasoning,    P_Correctness,
# Q_output,    Q_confidence,    Q_reasoning,    Q_Correctness,
# M_output_d1, M_confidence_d1, M_reasoning_d1, M_Correctness_d1, <- Round 1
# ...repeated for d2, d3, d4...
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
DEFAULT_DATASET        = "dataset.json"
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_RESULTS_JSON   = "codeqa_debate_results.json"
DEFAULT_RESULTS_CSV    = "codeqa_debate_results.csv"

NUM_DEBATE_ROUNDS      = 4

# Student models
MODEL_M = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_P = "microsoft/Phi-4-mini-instruct"
MODEL_Q = "Qwen/Qwen2.5-7B-Instruct"

# Judge model
MODEL_JUDGE = "Qwen/Qwen2.5-7B-Instruct"

CONTESTANT_MODELS = [
    ("M", MODEL_M),
    ("P", MODEL_P),
    ("Q", MODEL_Q),
]

# Display names used inside prompts
AGENT_DISPLAY_NAME = {
    "M": "Mistral-7B",
    "P": "Phi-4-mini",
    "Q": "Qwen2.5-7B",
}

# For each agent: its two peers  (label, display-name)
PEER_LABELS: Dict[str, List[Tuple[str, str]]] = {
    "M": [("P", "Phi-4-mini"),  ("Q", "Qwen2.5-7B")],
    "P": [("M", "Mistral-7B"),  ("Q", "Qwen2.5-7B")],
    "Q": [("M", "Mistral-7B"),  ("P", "Phi-4-mini")],
}


# ---------------------------------------------------------------------------
# Stopping criteria  -  halt once the first complete JSON object is generated
# ---------------------------------------------------------------------------
class StopAfterFirstJSONObject(StoppingCriteria):
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
            if esc:        esc = False;          continue
            if ch == "\\": esc = True;           continue
            if ch == '"':  in_str = not in_str;  continue
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


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_contestant_prompt(code: str, question: str) -> str:
    """Phase 0: independent QA with confidence score, no context from other agents."""
    return f"""You are given a Python code snippet and a question about it.

Task:
Read the code carefully and answer the question. Return your answer along with a brief explanation
and a confidence score reflecting how certain you are in your answer.

Definition of explanation:
- A short, high-level description of your reasoning. How did you arrive at the answer?
- Make it concise — 2-3 sentences.

Definition of confidence_score:
- A float between 0.0 and 1.0 representing how confident you are in your answer.
- 1.0 = completely certain, 0.5 = unsure, 0.0 = guessing.

Rules for predicted_output (your answer):
- Give a direct, concise answer to the question (e.g. "Yes", "No", "a suite", "an error").
- Match the expected answer style: short noun phrases or Yes/No where appropriate.
- Do not include the question or the code in your answer.

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "confidence_score": float,
  "explanation": string
}}

Code:
```python
{code}
```

Question:
{question}
"""


# (No teacher prompt - teacher has been removed from this pipeline)



def build_debate_prompt(
    code: str,
    question: str,
    my_label: str,
    round_num: int,
    my_output: str,
    my_confidence: float,
    my_reasoning: str,
    peer_responses: List[Tuple[str, str, float, str]],  # [(label, answer, confidence, reasoning), ...]
) -> str:
    """
    Student debate prompt for CodeQA - pure peer-debate with confidence scores.
    """
    peer_block = ""
    for peer_label, peer_output, peer_confidence, peer_reasoning in peer_responses:
        peer_block += f"""
--- Agent {peer_label} ({AGENT_DISPLAY_NAME[peer_label]}) ---
Answer: {peer_output}
Confidence: {peer_confidence:.2f}
Reasoning: {peer_reasoning}
"""

    return f"""You are Agent {my_label} ({AGENT_DISPLAY_NAME[my_label]}), participating in \
Round {round_num} of a {NUM_DEBATE_ROUNDS}-round multi-agent debate for code question answering.

You will receive:
1. The Python code snippet and the question about it.
2. YOUR answer, confidence score, and reasoning from the previous round.
3. The answers, confidence scores, and reasoning of the other two agents from the previous round.

Your job:
- Consider the other agents' reasoning and confidence scores critically — they may be right or wrong.
- A high-confidence peer answer deserves more weight, but do not follow the majority blindly.
- If another agent's reasoning exposes a flaw in your answer, revise it.
- If you remain confident in your answer, keep it and clearly explain why.
- Update your confidence score to reflect your current certainty after seeing all answers.

Rules for predicted_output (your answer):
- Give a direct, concise answer to the question (e.g. "Yes", "No", "a suite", "an error").
- Match the expected answer style: short noun phrases or Yes/No where appropriate.
- Do not include the question or the code in your answer.

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "confidence_score": float,
  "explanation": string
}}

=== Code ===
```python
{code}
```

=== Question ===
{question}

=== Your Answer from Previous Round (Agent {my_label}) ===
Answer: {my_output}
Confidence: {my_confidence:.2f}
Reasoning: {my_reasoning}

=== Other Agents' Answers from Previous Round ===
{peer_block}
Now carefully re-analyse the code and return your (possibly revised) answer as valid JSON.
"""


def build_judge_prompt(code: str, question: str, gold: str, prediction: str) -> str:
    return f"""You are an automated judge for a code question-answering task.

Your job is to decide whether the predicted answer is CORRECT compared to the gold answer.

Rules:
- The MEANING must match - focus on whether the predicted answer conveys the same information.
- Minor wording differences are ACCEPTABLE: "a suite" and "suite" or "Yes" and "yes" are CORRECT.
- For Yes/No questions, any clear affirmative ("Yes", "yeah", "correct", "true") counts as Yes,
  and any clear negative ("No", "nope", "false", "not really") counts as No.
- Partial answers that capture the key fact are CORRECT.
- Only mark INCORRECT when the meaning clearly differs from the gold answer.
- Return ONLY valid JSON with exactly these keys:
{{
  "verdict": "CORRECT" or "INCORRECT"
}}

Code:
```python
{code}
```

Question:
{question}

Gold Answer:
{gold}

Predicted Answer:
{prediction}
"""


# ---------------------------------------------------------------------------
# Model wrapper  (loads once, generates many times, then unloads)
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
# Dataset loader
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> List[Dict]:
    """
    Reads a CodeQA JSON dataset (list of dicts with keys: code, question, answer).
    Normalises to internal keys: code, input (=question), output (=answer).
    Also supports CSV with an optional 'id' column.
    """
    if path.lower().endswith(".csv"):
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append({
                    "id":     row.get("id", f"idx_{i}"),
                    "code":   row.get("code",   ""),
                    # CodeQA CSV: question / answer  OR  input / output
                    "input":  row.get("question", row.get("input",  "")),
                    "output": row.get("answer",   row.get("output", row.get("gold_answer", ""))),
                })
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of examples.")

    # Normalise CodeQA JSON fields to internal convention
    normalised = []
    for i, ex in enumerate(data):
        normalised.append({
            "id":     ex.get("id", f"idx_{i}"),
            "code":   ex.get("code",  ""),
            "input":  ex.get("question", ex.get("input",  "")),   # CodeQA uses "question"
            "output": ex.get("answer",   ex.get("output", "")),   # CodeQA uses "answer"
        })
    return normalised


# ---------------------------------------------------------------------------
# Phase 0 - baseline: each agent independently predicts the output
# ---------------------------------------------------------------------------
def run_contestant(
    label: str,
    model_name: str,
    dataset: List[Dict],
    max_new_tokens: int,
) -> List[Dict]:
    """
    Returns a list of per-example dicts:
      { "output": str, "confidence": float, "reasoning": str, "raw_text": str }
    """
    print(f"\n{'='*60}")
    print(f"PHASE 0 (Baseline) - Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        code   = ex.get("code",  "")
        inp    = ex.get("input", "")   # normalised from "question"
        ex_id  = ex.get("id",    f"idx_{idx}")
        prompt = build_contestant_prompt(code, inp)

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj    = _first_json_object(raw)
        out    = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        expl   = str(obj.get("explanation", "") or "") if obj else ""
        conf   = float(obj.get("confidence_score", 0.5)) if obj else 0.5
        conf   = max(0.0, min(1.0, conf))   # clamp to [0, 1]

        results.append({"output": out, "confidence": conf, "reasoning": expl, "raw_text": raw})
        print(f"-> predicted: {out[:60]!r}  conf={conf:.2f}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Debate round - each agent revises its answer guided by peers + confidence
# ---------------------------------------------------------------------------
def run_debate_round(
    label: str,
    model_name: str,
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    round_num: int,
    max_new_tokens: int,
) -> List[Dict]:
    """
    One pure peer-debate round for a single agent.

    Each agent sees its own previous answer + confidence score, plus both
    peers' previous answers + confidence scores, then revises freely.

    Returns a list of per-example dicts:
      { "output": str, "confidence": float, "reasoning": str, "raw_text": str }
    """
    print(f"\n{'='*60}")
    print(f"DEBATE ROUND {round_num}/{NUM_DEBATE_ROUNDS} - Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        code  = ex.get("code",  "")
        inp   = ex.get("input", "")
        ex_id = ex.get("id",    f"idx_{idx}")

        # This agent's own answer + confidence from the previous round
        my_output    = prev_round_preds[label][idx]["output"]
        my_confidence = prev_round_preds[label][idx]["confidence"]
        my_reasoning = prev_round_preds[label][idx]["reasoning"]

        # Both peers' answers + confidence from the previous round
        peer_responses: List[Tuple[str, str, float, str]] = []
        for peer_label, peer_name in PEER_LABELS[label]:
            peer_output    = prev_round_preds[peer_label][idx]["output"]
            peer_confidence = prev_round_preds[peer_label][idx]["confidence"]
            peer_reasoning = prev_round_preds[peer_label][idx]["reasoning"]
            peer_responses.append((peer_label, peer_output, peer_confidence, peer_reasoning))

        prompt = build_debate_prompt(
            code, inp,
            my_label=label,
            round_num=round_num,
            my_output=my_output,
            my_confidence=my_confidence,
            my_reasoning=my_reasoning,
            peer_responses=peer_responses,
        )

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj    = _first_json_object(raw)
        out    = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        expl   = str(obj.get("explanation", "") or "") if obj else ""
        conf   = float(obj.get("confidence_score", 0.5)) if obj else 0.5
        conf   = max(0.0, min(1.0, conf))   # clamp to [0, 1]

        # Fallback: retain previous answer if model returns nothing useful
        if not out.strip():
            out  = my_output
            conf = my_confidence
            expl = f"[round-{round_num} fallback] {my_reasoning}"

        results.append({"output": out, "confidence": conf, "reasoning": expl, "raw_text": raw})
        print(f"-> revised: {out[:60]!r}  conf={conf:.2f}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Judge - scores one full set of agent predictions
# ---------------------------------------------------------------------------
def run_judge(
    dataset: List[Dict],
    predictions: Dict[str, List[Dict]],
    phase_label: str = "",
    max_new_tokens: int = 128,
) -> Dict[str, List[str]]:
    """
    Evaluates every agent's predictions for a given phase/round.

    predictions  - { agent_label: [ {output, reasoning, raw_text}, ... ] }
    phase_label  - human-readable tag for logs (e.g. "Phase 0", "Round 3")

    Returns { agent_label: [ "CORRECT" | "INCORRECT", ... ] }
    """
    print(f"\n{'='*60}")
    print(f"JUDGE  [{phase_label}]  model={MODEL_JUDGE}")
    print(f"{'='*60}")

    runner      = ModelRunner(MODEL_JUDGE)
    correctness: Dict[str, List[str]] = {lbl: [] for lbl in predictions}

    for idx, ex in enumerate(dataset):
        code  = ex.get("code",   "")
        inp   = ex.get("input",  "")   # normalised from "question"
        gold  = ex.get("output", "")   # normalised from "answer"
        ex_id = ex.get("id",     f"idx_{idx}")

        print(f"  [Judge] Example {idx+1}/{len(dataset)}  id={ex_id}")

        for label in predictions:
            pred = predictions[label][idx]["output"]

            # Empty-output guard
            if not pred.strip():
                gold_is_empty = gold.strip() in ("''", '""', '')
                verdict = "CORRECT" if gold_is_empty else "INCORRECT"
                tag     = "(empty pred, gold also empty)" if gold_is_empty else "(empty pred)"
                print(f"    [{label}] pred=''  ->  {verdict}  {tag}")
                correctness[label].append(verdict)
                continue

            prompt  = build_judge_prompt(code, inp, gold, pred)
            raw     = runner.generate(prompt, max_new_tokens)

            obj     = _first_json_object(raw)
            verdict = (obj.get("verdict", "") or "").upper() if obj else ""
            if verdict not in ("CORRECT", "INCORRECT"):
                verdict = "INCORRECT"   # safe default on parse failure

            correctness[label].append(verdict)
            print(f"    [{label}] pred={pred[:40]!r}  ->  {verdict}")

    runner.unload()
    return correctness


# ---------------------------------------------------------------------------
# Output writers - all phases in one JSON + CSV
# ---------------------------------------------------------------------------
def write_outputs(
    dataset: List[Dict],
    all_preds:    List[Dict[str, List[Dict]]],   # index 0=Phase0, 1..4=rounds
    all_corr:     List[Dict[str, List[str]]],    # parallel correctness
    json_path: str = DEFAULT_RESULTS_JSON,
    csv_path:  str = DEFAULT_RESULTS_CSV,
) -> None:
    """
    Writes a single JSON + CSV with all phases side-by-side.

    Column naming:
      Phase 0 (baseline) : {lbl}_output,    {lbl}_confidence,    {lbl}_reasoning,    {lbl}_Correctness
      Debate round N     : {lbl}_output_dN, {lbl}_confidence_dN, {lbl}_reasoning_dN, {lbl}_Correctness_dN
    """
    labels = ["M", "P", "Q"]
    rows   = []

    for idx, ex in enumerate(dataset):
        row = {
            "code":        ex.get("code",   ""),
            "question":    ex.get("input",  ""),
            "gold_answer": ex.get("output", ""),
        }

        # Student predictions + confidence + correctness for every phase
        for phase_idx, (preds, corr) in enumerate(zip(all_preds, all_corr)):
            suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
            for lbl in labels:
                p = preds.get(lbl, [])
                c = corr.get(lbl,  [])
                row[f"{lbl}_output{suffix}"]      = p[idx]["output"]      if idx < len(p) else ""
                row[f"{lbl}_confidence{suffix}"]  = p[idx]["confidence"]  if idx < len(p) else ""
                row[f"{lbl}_reasoning{suffix}"]   = p[idx]["reasoning"]   if idx < len(p) else ""
                row[f"{lbl}_Correctness{suffix}"] = c[idx]                if idx < len(c) else ""

        rows.append(row)

    # ---- JSON ---------------------------------------------------------------
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved -> {json_path}")

    # ---- CSV ----------------------------------------------------------------
    fieldnames = ["code", "question", "gold_answer"]
    for phase_idx in range(len(all_preds)):
        suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
        for lbl in labels:
            fieldnames += [
                f"{lbl}_output{suffix}",
                f"{lbl}_confidence{suffix}",
                f"{lbl}_reasoning{suffix}",
                f"{lbl}_Correctness{suffix}",
            ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV results saved  -> {csv_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(
    all_corr: List[Dict[str, List[str]]],
    labels: List[str] = ("M", "P", "Q"),
) -> None:
    num_phases  = len(all_corr)
    def _round_label(r):
        return "Phase-0 (base)" if r == 0 else f"Round-{r}"
    phase_names = [_round_label(r) for r in range(num_phases)]
    col_w       = 22

    sep = "=" * (8 + col_w * num_phases)
    print("\n" + sep)
    print("SUMMARY  (peer debate with confidence scores  -  CodeQA)")
    print(sep)
    header = f"{'Agent':<8}" + "".join(f"{n:>{col_w}}" for n in phase_names)
    print(header)
    print("-" * (8 + col_w * num_phases))

    for lbl in labels:
        row_str  = f"  [{lbl}] "
        prev_pct = None
        for corr in all_corr:
            verdicts = corr.get(lbl, [])
            total    = len(verdicts)
            correct  = sum(1 for v in verdicts if v == "CORRECT")
            pct      = correct / total * 100 if total else 0.0

            if prev_pct is None:
                delta_str = ""
            else:
                diff  = pct - prev_pct
                arrow = "UP" if diff > 0 else ("DN" if diff < 0 else "--")
                delta_str = f"[{arrow}{abs(diff):.1f}%]"

            cell    = f"{correct}/{total}({pct:.1f}%) {delta_str}"
            row_str += f"{cell:>{col_w}}"
            prev_pct = pct

        print(row_str)

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dataset_path   = DEFAULT_DATASET
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    print(f"Dataset        : {dataset_path}  [CodeQA]")
    print(f"Debate rounds  : {NUM_DEBATE_ROUNDS}  (pure peer-debate, no teacher)")
    dataset = load_dataset(dataset_path)
    print(f"Loaded         : {len(dataset)} examples\n")

    # Accumulators for all phases
    all_preds: List[Dict[str, List[Dict]]] = []
    all_corr:  List[Dict[str, List[str]]]  = []

    # -----------------------------------------------------------------------
    # PHASE 0 - Baseline: each agent predicts independently with confidence
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# PHASE 0 - BASELINE (independent answers + confidence scores)")
    print("#" * 60)

    phase0_preds: Dict[str, List[Dict]] = {}
    for label, model_name in CONTESTANT_MODELS:
        phase0_preds[label] = run_contestant(label, model_name, dataset, max_new_tokens)

    phase0_corr = run_judge(
        dataset, phase0_preds,
        phase_label="Phase 0 (baseline)",
        max_new_tokens=128,
    )

    all_preds.append(phase0_preds)
    all_corr.append(phase0_corr)

    # -----------------------------------------------------------------------
    # ROUNDS 1 - NUM_DEBATE_ROUNDS  (pure peer-debate, no teacher)
    #
    # Each round:
    #   A) Each agent sees its own + peers' previous answers & confidence scores
    #      -> revises its answer and emits an updated confidence score
    #   B) Judge scores
    # -----------------------------------------------------------------------
    prev_preds = phase0_preds

    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        print("\n" + "#" * 60)
        print(f"# DEBATE ROUND {round_num} / {NUM_DEBATE_ROUNDS}  [PURE PEER DEBATE]")
        print("#" * 60)

        round_preds: Dict[str, List[Dict]] = {}
        for label, model_name in CONTESTANT_MODELS:
            round_preds[label] = run_debate_round(
                label, model_name, dataset,
                prev_round_preds=prev_preds,
                round_num=round_num,
                max_new_tokens=max_new_tokens,
            )

        round_corr = run_judge(
            dataset, round_preds,
            phase_label=f"Round {round_num} (pure debate)",
            max_new_tokens=128,
        )

        all_preds.append(round_preds)
        all_corr.append(round_corr)

        prev_preds = round_preds

    # -----------------------------------------------------------------------
    # Save all phases to JSON + CSV
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# SAVING RESULTS (all phases)")
    print("#" * 60)

    write_outputs(
        dataset,
        all_preds,
        all_corr,
        json_path=DEFAULT_RESULTS_JSON,
        csv_path=DEFAULT_RESULTS_CSV,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_summary(all_corr)


if __name__ == "__main__":
    main()