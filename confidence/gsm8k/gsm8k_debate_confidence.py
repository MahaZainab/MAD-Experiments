# =============================================================================
# GSM8K Multi-Model Evaluator  -  Peer Debate with Confidence Scores
#
# Dataset : GSM8K  (each example has at least: question, answer)
#           The task is to solve a grade-school math word problem and return
#           the final numeric answer.
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
#               plus both peers' previous answer values + confidence scores
#               ONLY (no reasoning shared — agents must think independently).
#               Agent then revises its answer and emits an updated confidence score.
#     Step B  : Judge scores revised answers.
#
# Output    : Single JSON + CSV with ALL rounds side-by-side.
# =============================================================================

import csv
import json
import re
from decimal import Decimal, InvalidOperation
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
DEFAULT_DATASET        = "dataset_gsm.json"           # GSM8K JSON / JSONL file
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_RESULTS_JSON   = "gsm8k_confidence_results.json"
DEFAULT_RESULTS_CSV    = "gsm8k_confidence_results.csv"

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

AGENT_DISPLAY_NAME = {
    "M": "Mistral-7B",
    "P": "Phi-4-mini",
    "Q": "Qwen2.5-7B",
}

PEER_LABELS: Dict[str, List[Tuple[str, str]]] = {
    "M": [("P", "Phi-4-mini"), ("Q", "Qwen2.5-7B")],
    "P": [("M", "Mistral-7B"), ("Q", "Qwen2.5-7B")],
    "Q": [("M", "Mistral-7B"), ("P", "Phi-4-mini")],
}


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------
class StopAfterFirstJSONObject(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.prompt_len:]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "{" not in text:
            return False
        start = text.find("{")
        sub = text[start:]
        depth, in_str, esc = 0, False, False
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


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _first_json_object(text: str) -> Optional[dict]:
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
                obj = json.loads(t[i:j + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def extract_gold_answer(answer_text: str) -> str:
    """Extract the GSM8K final answer, usually after '####'."""
    if answer_text is None:
        return ""
    text = str(answer_text).strip()
    if not text:
        return ""

    m = re.search(r"####\s*([^\n\r]+)", text)
    if m:
        return normalize_numeric_text(m.group(1))
    return normalize_numeric_text(text)


def normalize_numeric_text(text: str) -> str:
    """
    Normalise a numeric answer string so logically equivalent surface forms match.
    Examples: '$1,200.00' -> '1200', '018' -> '18', '3.50' -> '3.5'
    """
    if text is None:
        return ""

    t = str(text).strip()
    if not t:
        return ""

    t = t.replace(",", "")
    t = t.replace("$", "")
    t = t.replace("%", "")
    t = t.strip()

    # Prefer an explicit number if extra words are present.
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    if nums:
        t = nums[-1]

    try:
        d = Decimal(t)
        normalized = format(d.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized if normalized else "0"
    except (InvalidOperation, ValueError):
        return t


def extract_prediction_answer(raw_output: str, obj: Optional[dict]) -> str:
    if obj:
        pred = obj.get("predicted_output", obj.get("output", ""))
        if pred is not None:
            norm = normalize_numeric_text(str(pred))
            if norm:
                return norm

    if raw_output:
        m = re.search(r"####\s*([^\n\r]+)", raw_output)
        if m:
            return normalize_numeric_text(m.group(1))
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw_output.replace(",", ""))
        if nums:
            return normalize_numeric_text(nums[-1])

    return ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_contestant_prompt(question: str) -> str:
    return f"""You are solving a GSM8K grade-school math word problem.

Solve the problem carefully, but return ONLY the final numeric answer and a confidence score.
Do not include units, explanation, equations, or extra text in predicted_output.

Definition of confidence_score:
- A float between 0.0 and 1.0 representing how confident you are in your answer.
- 1.0 = completely certain, 0.5 = unsure, 0.0 = guessing.

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "confidence_score": float
}}

Question:
{question}
"""


def build_debate_prompt(
    question: str,
    my_label: str,
    round_num: int,
    my_output: str,
    my_confidence: float,
    peer_responses: List[Tuple[str, str, float]],
) -> str:
    peer_block = ""
    for peer_label, peer_output, peer_confidence in peer_responses:
        peer_block += (
            f"  Agent {peer_label} ({AGENT_DISPLAY_NAME[peer_label]}): "
            f"Answer={peer_output}  Confidence={peer_confidence:.2f}\n"
        )

    return f"""You are Agent {my_label} ({AGENT_DISPLAY_NAME[my_label]}), participating in Round {round_num}
of a {NUM_DEBATE_ROUNDS}-round peer debate for GSM8K math word problems.

You will receive:
1. The math word problem.
2. YOUR answer and confidence from the previous round.
3. The answer values and confidence scores of the other two agents (NO reasoning — you must think independently).

Your job:
- Re-solve the problem independently.
- Consider peers' answers and confidence as signals only.
- A high-confidence peer answer deserves attention, but do not blindly follow it.
- If peer signals make you doubt your answer, revise it.
- Update your confidence score to reflect your current certainty.

Rules for predicted_output:
- Return ONLY the final numeric answer as a string.
- No units, no explanation, no equations, no commas.

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "confidence_score": float
}}

=== Question ===
{question}

=== Your Answer from Previous Round ===
  Agent {my_label}: Answer={my_output}  Confidence={my_confidence:.2f}

=== Peers' Answers from Previous Round (answer + confidence only) ===
{peer_block}
Now reconsider the problem and return your (possibly revised) answer as valid JSON.
"""


def build_judge_prompt(question: str, gold: str, prediction: str) -> str:
    return f"""You are an automated judge for GSM8K.

Decide whether the predicted final numeric answer is equivalent to the gold final numeric answer.
Treat numerically equivalent forms as CORRECT (for example, 18 and 18.0).
Ignore commas and dollar signs. Return ONLY valid JSON with exactly these keys:
{{
  "verdict": "CORRECT" or "INCORRECT"
}}

Question:
{question}

Gold Final Answer:
{gold}

Predicted Final Answer:
{prediction}
"""


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, model_name: str):
        print(f"  Loading: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f"  Loaded:  {model_name}")

    def generate(self, prompt: str, max_new_tokens: int) -> str:
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
        gen_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def unload(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> List[Dict]:
    """
    Supports:
      - JSONL: one object per line
      - JSON array
      - CSV with columns including: question, answer, optionally id

    Normalises to keys:
      id, input (= question), output (= extracted final answer), answer_text (= full GSM8K rationale)
    """
    rows: List[Dict] = []

    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                full_answer = row.get("answer", row.get("output", ""))
                rows.append({
                    "id": row.get("id", f"idx_{i}"),
                    "input": row.get("question", row.get("input", "")),
                    "answer_text": full_answer,
                    "output": extract_gold_answer(full_answer),
                })
        return rows

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    raw_list = None

    # Try JSON array first
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            raw_list = parsed
        elif isinstance(parsed, dict):
            raw_list = [parsed]
    except Exception:
        raw_list = None

    # Fallback: JSONL
    if raw_list is None:
        raw_list = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            raw_list.append(json.loads(line))

    for i, ex in enumerate(raw_list):
        full_answer = ex.get("answer", ex.get("output", ""))
        rows.append({
            "id": ex.get("id", f"idx_{i}"),
            "input": ex.get("question", ex.get("input", "")),
            "answer_text": full_answer,
            "output": extract_gold_answer(full_answer),
        })

    return rows


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------
def run_contestant(label: str, model_name: str, dataset: List[Dict], max_new_tokens: int) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"PHASE 0 (Baseline) - Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        question = ex.get("input", "")
        ex_id = ex.get("id", f"idx_{idx}")
        prompt = build_contestant_prompt(question)

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj = _first_json_object(raw)
        out = extract_prediction_answer(raw, obj)
        conf = float(obj.get("confidence_score", 0.5)) if obj else 0.5
        conf = max(0.0, min(1.0, conf))

        results.append({"output": out, "confidence": conf, "raw_text": raw})
        print(f"-> predicted: {out!r}  conf={conf:.2f}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Debate rounds
# ---------------------------------------------------------------------------
def run_debate_round(
    label: str,
    model_name: str,
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    round_num: int,
    max_new_tokens: int,
) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"DEBATE ROUND {round_num}/{NUM_DEBATE_ROUNDS} - Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        question = ex.get("input", "")
        ex_id = ex.get("id", f"idx_{idx}")

        my_output = prev_round_preds[label][idx]["output"]
        my_confidence = prev_round_preds[label][idx]["confidence"]

        peer_responses: List[Tuple[str, str, float]] = []
        for peer_label, _ in PEER_LABELS[label]:
            peer_output = prev_round_preds[peer_label][idx]["output"]
            peer_confidence = prev_round_preds[peer_label][idx]["confidence"]
            peer_responses.append((peer_label, peer_output, peer_confidence))

        prompt = build_debate_prompt(
            question=question,
            my_label=label,
            round_num=round_num,
            my_output=my_output,
            my_confidence=my_confidence,
            peer_responses=peer_responses,
        )

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj = _first_json_object(raw)
        out = extract_prediction_answer(raw, obj)
        conf = float(obj.get("confidence_score", 0.5)) if obj else 0.5
        conf = max(0.0, min(1.0, conf))

        if not out:
            out = my_output
            conf = my_confidence

        results.append({"output": out, "confidence": conf, "raw_text": raw})
        print(f"-> revised: {out!r}  conf={conf:.2f}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
def run_judge(
    dataset: List[Dict],
    predictions: Dict[str, List[Dict]],
    phase_label: str = "",
    max_new_tokens: int = 128,
) -> Dict[str, List[str]]:
    print(f"\n{'='*60}")
    print(f"JUDGE  [{phase_label}]  model={MODEL_JUDGE}")
    print(f"{'='*60}")

    runner = ModelRunner(MODEL_JUDGE)
    correctness: Dict[str, List[str]] = {lbl: [] for lbl in predictions}

    for idx, ex in enumerate(dataset):
        question = ex.get("input", "")
        gold = normalize_numeric_text(ex.get("output", ""))
        ex_id = ex.get("id", f"idx_{idx}")

        print(f"  [Judge] Example {idx+1}/{len(dataset)}  id={ex_id}")

        for lbl in predictions:
            pred = normalize_numeric_text(predictions[lbl][idx]["output"])

            # Fast deterministic shortcut
            if pred and gold and pred == gold:
                verdict = "CORRECT"
                print(f"    [{lbl}] pred={pred!r}  ->  {verdict}  (normalized exact match)")
                correctness[lbl].append(verdict)
                continue

            if not pred:
                verdict = "INCORRECT"
                print(f"    [{lbl}] pred=''  ->  {verdict}  (empty pred)")
                correctness[lbl].append(verdict)
                continue

            prompt = build_judge_prompt(question, gold, pred)
            raw = runner.generate(prompt, max_new_tokens)
            obj = _first_json_object(raw)
            verdict = (obj.get("verdict", "") or "").upper() if obj else ""
            if verdict not in ("CORRECT", "INCORRECT"):
                verdict = "INCORRECT"

            correctness[lbl].append(verdict)
            print(f"    [{lbl}] pred={pred!r}  gold={gold!r}  ->  {verdict}")

    runner.unload()
    return correctness


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_outputs(
    dataset: List[Dict],
    all_preds: List[Dict[str, List[Dict]]],
    all_corr: List[Dict[str, List[str]]],
    json_path: str = DEFAULT_RESULTS_JSON,
    csv_path: str = DEFAULT_RESULTS_CSV,
) -> None:
    labels = ["M", "P", "Q"]
    rows = []

    for idx, ex in enumerate(dataset):
        row = {
            "id": ex.get("id", ""),
            "question": ex.get("input", ""),
            "gold_answer": ex.get("output", ""),
            "gold_rationale": ex.get("answer_text", ""),
        }

        for phase_idx, (preds, corr) in enumerate(zip(all_preds, all_corr)):
            suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
            for lbl in labels:
                p = preds.get(lbl, [])
                c = corr.get(lbl, [])
                row[f"{lbl}_output{suffix}"] = p[idx]["output"] if idx < len(p) else ""
                row[f"{lbl}_confidence{suffix}"] = p[idx]["confidence"] if idx < len(p) else ""
                row[f"{lbl}_Correctness{suffix}"] = c[idx] if idx < len(c) else ""

        rows.append(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved -> {json_path}")

    fieldnames = ["id", "question", "gold_answer", "gold_rationale"]
    for phase_idx in range(len(all_preds)):
        suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
        for lbl in labels:
            fieldnames += [
                f"{lbl}_output{suffix}",
                f"{lbl}_confidence{suffix}",
                f"{lbl}_Correctness{suffix}",
            ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV results saved  -> {csv_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(all_corr: List[Dict[str, List[str]]], labels: List[str] = ("M", "P", "Q")) -> None:
    num_phases = len(all_corr)

    def _round_label(r: int) -> str:
        return "Phase-0 (base)" if r == 0 else f"Round-{r}"

    phase_names = [_round_label(r) for r in range(num_phases)]
    col_w = 22

    sep = "=" * (8 + col_w * num_phases)
    print("\n" + sep)
    print("SUMMARY  (peer debate, answer+confidence only  -  GSM8K)")
    print(sep)
    header = f"{'Agent':<8}" + "".join(f"{n:>{col_w}}" for n in phase_names)
    print(header)
    print("-" * (8 + col_w * num_phases))

    for lbl in labels:
        row_str = f"  [{lbl}] "
        prev_pct = None
        for corr in all_corr:
            verdicts = corr.get(lbl, [])
            total = len(verdicts)
            correct = sum(1 for v in verdicts if v == "CORRECT")
            pct = correct / total * 100 if total else 0.0

            if prev_pct is None:
                delta_str = ""
            else:
                diff = pct - prev_pct
                arrow = "UP" if diff > 0 else ("DN" if diff < 0 else "--")
                delta_str = f"[{arrow}{abs(diff):.1f}%]"

            cell = f"{correct}/{total}({pct:.1f}%) {delta_str}"
            row_str += f"{cell:>{col_w}}"
            prev_pct = pct

        print(row_str)

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dataset_path = DEFAULT_DATASET
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    print(f"Dataset        : {dataset_path}  [GSM8K]")
    print(f"Debate rounds  : {NUM_DEBATE_ROUNDS}  (pure peer-debate, answer+confidence only)")
    dataset = load_dataset(dataset_path)
    print(f"Loaded         : {len(dataset)} examples\n")

    all_preds: List[Dict[str, List[Dict]]] = []
    all_corr: List[Dict[str, List[str]]] = []

    print("\n" + "#" * 60)
    print("# PHASE 0 - BASELINE (independent answers + confidence scores)")
    print("#" * 60)

    phase0_preds: Dict[str, List[Dict]] = {}
    for label, model_name in CONTESTANT_MODELS:
        phase0_preds[label] = run_contestant(label, model_name, dataset, max_new_tokens)

    phase0_corr = run_judge(
        dataset,
        phase0_preds,
        phase_label="Phase 0 (baseline)",
        max_new_tokens=128,
    )

    all_preds.append(phase0_preds)
    all_corr.append(phase0_corr)

    prev_preds = phase0_preds
    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        print("\n" + "#" * 60)
        print(f"# DEBATE ROUND {round_num} / {NUM_DEBATE_ROUNDS}  [PURE PEER DEBATE]")
        print("#" * 60)

        round_preds: Dict[str, List[Dict]] = {}
        for label, model_name in CONTESTANT_MODELS:
            round_preds[label] = run_debate_round(
                label,
                model_name,
                dataset,
                prev_round_preds=prev_preds,
                round_num=round_num,
                max_new_tokens=max_new_tokens,
            )

        round_corr = run_judge(
            dataset,
            round_preds,
            phase_label=f"Round {round_num} (pure debate)",
            max_new_tokens=128,
        )

        all_preds.append(round_preds)
        all_corr.append(round_corr)
        prev_preds = round_preds

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

    print_summary(all_corr)


if __name__ == "__main__":
    main()
