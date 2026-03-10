
import csv
import json
import re
import time
import requests
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET      = "dataset_gsm.json"
DEFAULT_RESULTS_JSON = "gsm8k_confidence_results.json"
DEFAULT_RESULTS_CSV  = "gsm8k_confidence_results.csv"

NUM_DEBATE_ROUNDS    = 4

# Ollama server (change host/port if needed)
OLLAMA_BASE_URL      = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Model names exactly as registered in Ollama
# ---------------------------------------------------------------------------
MODEL_M     = "mistral-small:24b"
MODEL_P     = "phi4:14b"
MODEL_Q     = "qwen3.5:35b"
MODEL_JUDGE = "qwen2.5:7b"

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
    "M": [("P", "Phi-4-mini"),  ("Q", "Qwen2.5-7B")],
    "P": [("M", "Mistral-7B"),  ("Q", "Qwen2.5-7B")],
    "Q": [("M", "Mistral-7B"),  ("P", "Phi-4-mini")],
}


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------
class OllamaRunner:
    """
    Thin wrapper around Ollama /api/generate.
    No GPU memory management needed — Ollama handles that.
    """

    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL):
        self.model    = model_name
        self.base_url = base_url.rstrip("/")
        self._check_model()

    def _check_model(self) -> None:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            names = [m["name"] for m in resp.json().get("models", [])]
            norm  = lambda n: n.split(":")[0]
            if not any(norm(n) == norm(self.model) or n == self.model for n in names):
                print(
                    f"  [WARNING] Model '{self.model}' not found locally.\n"
                    f"  Run:  ollama pull {self.model}"
                )
            else:
                print(f"  [OK] Model ready: {self.model}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}.\n"
                "Make sure Ollama is running:  ollama serve"
            )

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict":    max_tokens,
                "temperature":    0.0,
                "top_p":          1.0,
                "repeat_penalty": 1.0,
            },
        }
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
            except requests.exceptions.Timeout:
                print(f"  [WARN] Timeout on attempt {attempt+1}/3, retrying...")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"  [ERROR] Ollama request failed: {e}")
                return ""
        return ""


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
                obj = json.loads(t[i : j + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def _safe_conf(obj: Optional[dict], fallback: float = 0.5) -> float:
    if not obj:
        return fallback
    try:
        return max(0.0, min(1.0, float(obj.get("confidence_score", fallback))))
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# GSM8K gold answer extractor
# ---------------------------------------------------------------------------
def extract_gold_answer(raw_answer: str) -> str:
    """
    GSM8K answers end with '#### <number>'.
    Returns the numeric string after ####, stripped of commas.
    Falls back to the full answer string if the pattern is not found.
    """
    m = re.search(r"####\s*([\d,\.\-]+)", raw_answer)
    if m:
        return m.group(1).replace(",", "").strip()
    return raw_answer.strip()


def _normalise_number(text: str) -> str:
    """Strip commas, leading zeros, and whitespace for numeric comparison."""
    t = text.strip().replace(",", "")
    try:
        # Normalise floats: 18.0 == 18
        f = float(t)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return t


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_contestant_prompt(question: str) -> str:
    """Phase 0: independent math solving with confidence score."""
    return f"""You are solving a grade-school math word problem.

Task:
Read the problem carefully, reason step by step, and give the final numeric answer.

Definition of confidence_score:
- A float between 0.0 and 1.0 representing how confident you are in your answer.
- 1.0 = completely certain, 0.5 = unsure, 0.0 = guessing.

Rules for predicted_output:
- Return ONLY the final numeric answer (e.g. 18, 70000, 3).
- Do not include units, currency symbols, or explanation — only the number.

Return ONLY valid JSON with exactly these two keys:
{{
  "predicted_output": string,
  "confidence_score": float
}}

Problem:
{question}
"""


def build_debate_prompt(
    question: str,
    my_label: str,
    round_num: int,
    my_output: str,
    my_confidence: float,
    peer_responses: List[Tuple[str, str, float]],   # (label, answer, confidence)
) -> str:
    """
    Peer-debate prompt for GSM8K.
    Peers share ONLY their numeric answer + confidence — no reasoning.
    """
    peer_block = ""
    for peer_label, peer_output, peer_confidence in peer_responses:
        peer_block += (
            f"  Agent {peer_label} ({AGENT_DISPLAY_NAME[peer_label]}): "
            f"Answer={peer_output}  Confidence={peer_confidence:.2f}\n"
        )

    return f"""You are Agent {my_label} ({AGENT_DISPLAY_NAME[my_label]}), in Round {round_num} \
of a {NUM_DEBATE_ROUNDS}-round peer debate for math problem solving.

You will receive:
1. The math word problem.
2. YOUR answer and confidence from the previous round.
3. Both peers' numeric answers and confidence scores ONLY (no reasoning — think independently).

Your job:
- Re-read and re-solve the problem carefully using your own reasoning.
- Consider what each peer answered and how confident they were.
- A high-confidence peer answer deserves more weight, but do not follow the majority blindly.
- Revise your answer if your re-calculation or peer signals give you reason to doubt it.
- Keep your answer if you remain confident after re-checking.
- Update your confidence score to reflect your current certainty.

Rules for predicted_output:
- Return ONLY the final numeric answer (e.g. 18, 70000, 3).
- Do not include units, currency symbols, or explanation — only the number.

Return ONLY valid JSON with exactly these two keys:
{{
  "predicted_output": string,
  "confidence_score": float
}}

=== Problem ===
{question}

=== Your Answer from Previous Round ===
  Agent {my_label}: Answer={my_output}  Confidence={my_confidence:.2f}

=== Peers' Answers from Previous Round (answer + confidence only) ===
{peer_block}
Now re-solve the problem carefully and return your (possibly revised) answer as valid JSON.
"""


def build_judge_prompt(question: str, gold: str, prediction: str) -> str:
    return f"""You are an automated judge for a math word problem task.

Decide whether the predicted answer matches the gold answer numerically.

Rules:
- CORRECT  if the predicted number equals the gold number (ignore commas, trailing zeros).
- INCORRECT otherwise.
- If the prediction contains extra text, extract the number from it and compare.
  If you cannot find a clear number, mark INCORRECT.

Return ONLY valid JSON:
{{
  "verdict": "CORRECT" or "INCORRECT"
}}

Problem:
{question}

Gold Answer:
{gold}

Predicted Answer:
{prediction}
"""


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> List[Dict]:
    """
    Reads a GSM8K JSON dataset.
    Supports:
      - JSON array : [{question, answer}, ...]
      - JSONL      : one object per line with the same keys

    Adds:
      - id          : auto-generated as idx_N if not present
      - gold_answer : numeric string extracted from after "####" in answer
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):           # JSON array
        raw_list = json.loads(content)
        if not isinstance(raw_list, list):
            raise ValueError("Dataset JSON must be a list of examples.")
    else:                                  # JSONL
        raw_list = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    raw_list.append(json.loads(line))
                except Exception:
                    pass

    rows = []
    for i, ex in enumerate(raw_list):
        raw_ans = ex.get("answer", "")
        rows.append({
            "id":          ex.get("id", f"idx_{i}"),
            "question":    ex.get("question", ""),
            "answer":      raw_ans,                       # full chain-of-thought answer
            "gold_answer": extract_gold_answer(raw_ans),  # numeric only
        })
    return rows


# ---------------------------------------------------------------------------
# Phase 0 — baseline
# ---------------------------------------------------------------------------
def run_contestant(
    label: str,
    model_name: str,
    dataset: List[Dict],
    max_tokens: int = 256,
) -> List[Dict]:
    """Returns [{output, confidence, raw_text}, ...]"""
    print(f"\n{'='*60}")
    print(f"PHASE 0 (Baseline) — Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = OllamaRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        ex_id = ex.get("id", f"idx_{idx}")
        print(f"  [{label}] {idx+1}/{len(dataset)}  id={ex_id}", end="  ")

        prompt = build_contestant_prompt(ex["question"])
        raw    = runner.generate(prompt, max_tokens)
        obj    = _first_json_object(raw)

        out  = str(obj.get("predicted_output", "") or "") if obj else ""
        conf = _safe_conf(obj)

        # Fallback: grab last number from raw output
        if not out.strip():
            nums = re.findall(r"[\d,]+\.?\d*", raw)
            out  = nums[-1].replace(",", "") if nums else ""

        results.append({"output": out.strip(), "confidence": conf, "raw_text": raw})
        print(f"-> {out!r}  conf={conf:.2f}")

    return results


# ---------------------------------------------------------------------------
# Debate round — pure peer debate, answer + confidence only
# ---------------------------------------------------------------------------
def run_debate_round(
    label: str,
    model_name: str,
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    round_num: int,
    max_tokens: int = 256,
) -> List[Dict]:
    """Returns [{output, confidence, raw_text}, ...]"""
    print(f"\n{'='*60}")
    print(f"DEBATE ROUND {round_num}/{NUM_DEBATE_ROUNDS} — Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = OllamaRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        ex_id = ex.get("id", f"idx_{idx}")

        my_output     = prev_round_preds[label][idx]["output"]
        my_confidence = prev_round_preds[label][idx]["confidence"]

        # Peers share numeric answer + confidence ONLY
        peer_responses: List[Tuple[str, str, float]] = [
            (
                pl,
                prev_round_preds[pl][idx]["output"],
                prev_round_preds[pl][idx]["confidence"],
            )
            for pl, _ in PEER_LABELS[label]
        ]

        prompt = build_debate_prompt(
            ex["question"],
            my_label=label,
            round_num=round_num,
            my_output=my_output,
            my_confidence=my_confidence,
            peer_responses=peer_responses,
        )

        print(f"  [{label}] {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw  = runner.generate(prompt, max_tokens)
        obj  = _first_json_object(raw)

        out  = str(obj.get("predicted_output", "") or "") if obj else ""
        conf = _safe_conf(obj, fallback=my_confidence)

        if not out.strip():
            nums = re.findall(r"[\d,]+\.?\d*", raw)
            out  = nums[-1].replace(",", "") if nums else ""

        # Fallback: retain previous answer
        if not out.strip():
            out  = my_output
            conf = my_confidence

        results.append({"output": out.strip(), "confidence": conf, "raw_text": raw})
        print(f"-> {out!r}  conf={conf:.2f}")

    return results


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
def run_judge(
    dataset: List[Dict],
    predictions: Dict[str, List[Dict]],
    phase_label: str = "",
    max_tokens: int = 64,
) -> Dict[str, List[str]]:
    """Returns {agent_label: ["CORRECT"|"INCORRECT", ...]}"""
    print(f"\n{'='*60}")
    print(f"JUDGE  [{phase_label}]  model={MODEL_JUDGE}")
    print(f"{'='*60}")

    runner      = OllamaRunner(MODEL_JUDGE)
    correctness: Dict[str, List[str]] = {lbl: [] for lbl in predictions}

    for idx, ex in enumerate(dataset):
        gold  = ex.get("gold_answer", "").strip()
        ex_id = ex.get("id", f"idx_{idx}")
        print(f"  [Judge] {idx+1}/{len(dataset)}  id={ex_id}")

        for lbl in predictions:
            pred = predictions[lbl][idx]["output"].strip()

            # Fast numeric exact-match shortcut
            if _normalise_number(pred) == _normalise_number(gold):
                correctness[lbl].append("CORRECT")
                print(f"    [{lbl}] -> CORRECT (exact match)")
                continue

            if not pred:
                verdict = "CORRECT" if not gold else "INCORRECT"
                correctness[lbl].append(verdict)
                print(f"    [{lbl}] (empty) -> {verdict}")
                continue

            raw     = runner.generate(
                build_judge_prompt(ex["question"], gold, pred),
                max_tokens,
            )
            obj     = _first_json_object(raw)
            verdict = (obj.get("verdict", "") or "").upper() if obj else ""
            if verdict not in ("CORRECT", "INCORRECT"):
                verdict = "INCORRECT"

            correctness[lbl].append(verdict)
            print(f"    [{lbl}] pred={pred!r}  gold={gold!r}  -> {verdict}")

    return correctness


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_outputs(
    dataset: List[Dict],
    all_preds: List[Dict[str, List[Dict]]],
    all_corr:  List[Dict[str, List[str]]],
    json_path: str = DEFAULT_RESULTS_JSON,
    csv_path:  str = DEFAULT_RESULTS_CSV,
) -> None:
    labels = ["M", "P", "Q"]
    rows   = []

    for idx, ex in enumerate(dataset):
        row = {
            "id":          ex.get("id",          ""),
            "question":    ex.get("question",    ""),
            "gold_answer": ex.get("gold_answer", ""),
        }
        for phase_idx, (preds, corr) in enumerate(zip(all_preds, all_corr)):
            suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
            for lbl in labels:
                p = preds.get(lbl, [])
                c = corr.get(lbl,  [])
                row[f"{lbl}_output{suffix}"]      = p[idx]["output"]     if idx < len(p) else ""
                row[f"{lbl}_confidence{suffix}"]  = p[idx]["confidence"] if idx < len(p) else ""
                row[f"{lbl}_Correctness{suffix}"] = c[idx]               if idx < len(c) else ""
        rows.append(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved -> {json_path}")

    fieldnames = ["id", "question", "gold_answer"]
    for phase_idx in range(len(all_preds)):
        suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
        for lbl in labels:
            fieldnames += [
                f"{lbl}_output{suffix}",
                f"{lbl}_confidence{suffix}",
                f"{lbl}_Correctness{suffix}",
            ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved  -> {csv_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(all_corr: List[Dict[str, List[str]]], labels=("M", "P", "Q")) -> None:
    phase_names = ["Phase-0 (base)"] + [f"Round-{r}" for r in range(1, len(all_corr))]
    col_w = 22
    sep   = "=" * (8 + col_w * len(all_corr))
    print(f"\n{sep}")
    print("SUMMARY  (peer debate, answer+confidence only  —  GSM8K)")
    print(sep)
    print(f"{'Agent':<8}" + "".join(f"{n:>{col_w}}" for n in phase_names))
    print("-" * (8 + col_w * len(all_corr)))

    for lbl in labels:
        row_str  = f"  [{lbl}] "
        prev_pct = None
        for corr in all_corr:
            verdicts = corr.get(lbl, [])
            total    = len(verdicts)
            correct  = sum(1 for v in verdicts if v == "CORRECT")
            pct      = correct / total * 100 if total else 0.0
            if prev_pct is None:
                delta = ""
            else:
                d     = pct - prev_pct
                arrow = "UP" if d > 0 else ("DN" if d < 0 else "--")
                delta = f"[{arrow}{abs(d):.1f}%]"
            row_str += f"{f'{correct}/{total}({pct:.1f}%) {delta}':>{col_w}}"
            prev_pct = pct
        print(row_str)

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("GSM8K — Peer Debate with Confidence (Ollama backend)")
    print("=" * 60)
    print(f"Ollama URL     : {OLLAMA_BASE_URL}")
    print(f"Student models : M={MODEL_M}  P={MODEL_P}  Q={MODEL_Q}")
    print(f"Judge model    : {MODEL_JUDGE}")
    print(f"Debate rounds  : {NUM_DEBATE_ROUNDS}  (pure peer-debate, answer+confidence only)")
    print(f"Dataset        : {DEFAULT_DATASET}")
    print()

    dataset = load_dataset(DEFAULT_DATASET)
    print(f"Loaded {len(dataset)} examples\n")

    all_preds: List[Dict[str, List[Dict]]] = []
    all_corr:  List[Dict[str, List[str]]]  = []

    # ------------------------------------------------------------------
    # PHASE 0 — baseline
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# PHASE 0 — BASELINE (independent answers + confidence)")
    print("#" * 60)

    phase0_preds: Dict[str, List[Dict]] = {}
    for label, model_name in CONTESTANT_MODELS:
        phase0_preds[label] = run_contestant(label, model_name, dataset)

    phase0_corr = run_judge(dataset, phase0_preds, phase_label="Phase 0 (baseline)")
    all_preds.append(phase0_preds)
    all_corr.append(phase0_corr)

    # ------------------------------------------------------------------
    # ROUNDS 1 – NUM_DEBATE_ROUNDS  (pure peer-debate)
    # ------------------------------------------------------------------
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
            )

        round_corr = run_judge(
            dataset, round_preds,
            phase_label=f"Round {round_num} (pure debate)",
        )
        all_preds.append(round_preds)
        all_corr.append(round_corr)
        prev_preds = round_preds

    # ------------------------------------------------------------------
    # Save + summarise
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# SAVING RESULTS")
    print("#" * 60)
    write_outputs(dataset, all_preds, all_corr)
    print_summary(all_corr)


if __name__ == "__main__":
    main()
