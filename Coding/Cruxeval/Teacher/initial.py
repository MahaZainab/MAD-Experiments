# =============================================================================
# CruxEval Multi-Model Evaluator  -  Teacher-Guided Theory-of-Mind Debate
#
# Models:
#   M       = Mistral-7B-Instruct-v0.3         (student agent)
#   P       = Phi-4-mini-instruct               (student agent)
#   Q       = Qwen2.5-7B-Instruct               (student agent)
#   Teacher = Qwen3-Coder-30B-A3B-Instruct      (teacher / ToM guide)
#   Judge   = Qwen2.5-7B-Instruct               (evaluator, unchanged)
#
# Pipeline
# --------
# Phase 0   : Each student independently predicts the output  (baseline)
#             --> Judge scores Phase 0
#
# Round 1-5 : For EACH round:
#   Step A  : Teacher reads ALL three students' previous answers + reasoning
#             and produces targeted, per-agent guidance using Theory-of-Mind
#             (tells each agent WHERE its reasoning may have gone wrong,
#              WITHOUT revealing the correct answer directly).
#   Step B  : Each student receives its own previous answer, both peers'
#             previous answers, AND the teacher's personal guidance for it.
#             It then revises its answer.
#   Step C  : Judge scores all three revised answers.
#
# Output    : Single JSON + CSV with ALL rounds side-by-side.
#
# CSV columns
# -----------
# code, input, gold_output,
# M_output,    M_reasoning,    M_Correctness,           <- Phase 0
# P_output,    P_reasoning,    P_Correctness,
# Q_output,    Q_reasoning,    Q_Correctness,
# M_output_d1, M_reasoning_d1, M_Correctness_d1,        <- Round 1
# P_output_d1, P_reasoning_d1, P_Correctness_d1,
# Q_output_d1, Q_reasoning_d1, Q_Correctness_d1,
# ...repeated for d2, d3, d4, d5...
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
DEFAULT_RESULTS_JSON   = "cruxeval_debate_results.json"
DEFAULT_RESULTS_CSV    = "cruxeval_debate_results.csv"

NUM_DEBATE_ROUNDS      = 5

# Student models
MODEL_M = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_P = "microsoft/Phi-4-mini-instruct"
MODEL_Q = "Qwen/Qwen2.5-7B-Instruct"

# Teacher model  (larger model that guides students via Theory-of-Mind)
MODEL_TEACHER = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

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
def build_contestant_prompt(code: str, inp: str) -> str:
    """Phase 0: independent prediction, no context from other agents."""
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


def build_teacher_prompt(
    code: str,
    inp: str,
    round_num: int,
    agent_responses: List[Tuple[str, str, str]],   # [(label, output, reasoning), ...]
) -> str:
    """
    Teacher (Theory-of-Mind) prompt.

    The teacher sees ALL three students' answers and must produce
    targeted, Socratic guidance for EACH student individually.
    It must NOT reveal the correct answer directly - only point out
    where each student's reasoning may have gone wrong and suggest
    what to re-examine.

    Returns JSON:
    {
      "guidance_M": "...",
      "guidance_P": "...",
      "guidance_Q": "..."
    }
    """
    agent_block = ""
    for label, output, reasoning in agent_responses:
        agent_block += f"""
--- Agent {label} ({AGENT_DISPLAY_NAME[label]}) ---
Answer: {output}
Reasoning: {reasoning}
"""

    return f"""You are an expert Python teacher and Theory-of-Mind reasoner participating in \
Round {round_num} of a {NUM_DEBATE_ROUNDS}-round multi-agent debate.

Three small language model agents (M, P, Q) are each trying to predict the output of a \
Python code snippet. You have received their latest answers and reasoning.

Your role - Theory-of-Mind guidance:
- Carefully trace through the code yourself to understand the correct output.
- For EACH agent, reason about WHY that agent may have arrived at its answer.
- Identify the specific step or assumption in each agent's reasoning that is flawed \
(or confirm it is on the right track).
- Write personalised, Socratic guidance for each agent:
  * Point out exactly where their reasoning may be going wrong.
  * Ask a targeted question or suggest a specific step to re-examine.
  * Do NOT directly reveal the correct answer.
  * Keep each guidance concise (2-4 sentences).

Return ONLY valid JSON with exactly these keys (one guidance string per agent):
{{
  "guidance_M": string,
  "guidance_P": string,
  "guidance_Q": string
}}

=== Code ===
```python
{code}
```

=== Input ===
{inp}

=== Agent Answers (Round {round_num - 1} output) ===
{agent_block}
"""


def build_debate_prompt(
    code: str,
    inp: str,
    my_label: str,
    round_num: int,
    my_output: str,
    my_reasoning: str,
    peer_responses: List[Tuple[str, str, str]],   # [(label, output, reasoning), ...]
    teacher_guidance: str,                         # personalised hint from the teacher
) -> str:
    """
    Student debate prompt with teacher guidance injected.

    Each student receives:
      - Original code + input
      - Its own previous answer + reasoning
      - Both peers' previous answers + reasoning
      - The teacher's personalised guidance (Theory-of-Mind hint)
    """
    peer_block = ""
    for peer_label, peer_output, peer_reasoning in peer_responses:
        peer_block += f"""
--- Agent {peer_label} ({AGENT_DISPLAY_NAME[peer_label]}) ---
Answer: {peer_output}
Reasoning: {peer_reasoning}
"""

    return f"""You are Agent {my_label} ({AGENT_DISPLAY_NAME[my_label]}), participating in \
Round {round_num} of a {NUM_DEBATE_ROUNDS}-round multi-agent debate for Python output prediction.

You will receive:
1. The Python code and its input.
2. YOUR answer and reasoning from the previous round (which may be correct or incorrect).
3. The answers and reasoning of the other two agents from the previous round.
4. Personalised guidance from the Teacher, who has analysed all agents' reasoning using \
Theory-of-Mind. The teacher will NOT give you the answer directly - but will point out \
where your reasoning might be going wrong.

Your job:
- Read the teacher's guidance carefully - it is targeted specifically at your reasoning.
- Trace through the code step by step, paying attention to the specific concern the teacher raised.
- Consider the other agents' reasoning critically - they may be right or wrong.
- Do NOT blindly follow the majority. A lone agent can be right while two are wrong.
- If the teacher's guidance or another agent's reasoning exposes a flaw in your answer, revise it.
- If you remain confident in your answer, keep it and clearly explain why.

Rules for predicted_output:
- Return the value exactly as Python would display it (repr-style).
- If it is a string, include quotes.
- If it contains backslash escapes like \\n or \\t, return them as literal characters \
(backslash+n), not as real newlines/tabs.

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

=== Your Answer from Previous Round (Agent {my_label}) ===
Answer: {my_output}
Reasoning: {my_reasoning}

=== Other Agents' Answers from Previous Round ===
{peer_block}

=== Teacher Guidance (personalised for Agent {my_label}) ===
{teacher_guidance}

Now carefully re-analyse the code using the teacher's guidance and return your \
(possibly revised) answer as valid JSON.
"""


def build_judge_prompt(code: str, inp: str, gold: str, prediction: str) -> str:
    return f"""You are an automated judge for a Python output prediction task.

Your job is to decide whether the predicted output is CORRECT compared to the gold output.

Rules:
- The UNDERLYING VALUE must match - focus on what the code actually returns.
- Quote-style differences are ACCEPTABLE: if gold is "'hello'" and prediction is
  "hello" (or vice versa), treat them as CORRECT - the underlying string value matches.
- Treat semantically equivalent representations as correct.
- Only mark INCORRECT when the actual computed value differs, not just its formatting.
- Return ONLY valid JSON with exactly these keys:
{{
  "verdict": "CORRECT" or "INCORRECT"
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
    Reads a JSON dataset (list of dicts with keys: code, input, output).
    Falls back to CSV if the path ends with .csv.
    """
    if path.lower().endswith(".csv"):
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

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of examples.")
    return data


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
      { "output": str, "reasoning": str, "raw_text": str }
    """
    print(f"\n{'='*60}")
    print(f"PHASE 0 (Baseline) - Agent [{label}]  {model_name}")
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
        print(f"-> predicted: {out[:60]!r}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Teacher pass - generates per-agent Theory-of-Mind guidance for one round
# ---------------------------------------------------------------------------
def run_teacher(
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    round_num: int,
    max_new_tokens: int = 512,
) -> List[Dict[str, str]]:
    """
    For each example, the teacher reads all three agents' latest answers and
    produces a personalised guidance string for each agent.

    Returns a list (one entry per example) of dicts:
      {
        "guidance_M": str,
        "guidance_P": str,
        "guidance_Q": str,
        "raw_text":   str,   # full teacher output for debugging
      }

    If the teacher fails to parse for a given example, all guidance strings
    default to a neutral fallback so the debate can continue.
    """
    print(f"\n{'='*60}")
    print(f"TEACHER PASS  Round {round_num}  model={MODEL_TEACHER}")
    print(f"{'='*60}")

    runner  = ModelRunner(MODEL_TEACHER)
    results = []

    for idx, ex in enumerate(dataset):
        code  = ex.get("code",  "")
        inp   = ex.get("input", "")
        ex_id = ex.get("id",    f"idx_{idx}")

        # Collect all three agents' latest answers
        agent_responses: List[Tuple[str, str, str]] = []
        for label, _ in CONTESTANT_MODELS:
            out  = prev_round_preds[label][idx]["output"]
            expl = prev_round_preds[label][idx]["reasoning"]
            agent_responses.append((label, out, expl))

        prompt = build_teacher_prompt(code, inp, round_num, agent_responses)

        print(f"  [Teacher] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj = _first_json_object(raw)

        # Extract per-agent guidance; fall back gracefully if parse fails
        fallback = (
            "Re-examine each step of the code carefully. "
            "Make sure you are tracing variable values correctly."
        )
        g_M = str(obj.get("guidance_M", fallback) or fallback) if obj else fallback
        g_P = str(obj.get("guidance_P", fallback) or fallback) if obj else fallback
        g_Q = str(obj.get("guidance_Q", fallback) or fallback) if obj else fallback

        results.append({
            "guidance_M": g_M,
            "guidance_P": g_P,
            "guidance_Q": g_Q,
            "raw_text":   raw,
        })
        print(f"-> guidance generated (M:{len(g_M)}c  P:{len(g_P)}c  Q:{len(g_Q)}c)")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Debate round - each agent revises its answer guided by teacher + peers
# ---------------------------------------------------------------------------
def run_debate_round(
    label: str,
    model_name: str,
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    teacher_guidance: List[Dict[str, str]],        # output of run_teacher()
    round_num: int,
    max_new_tokens: int,
) -> List[Dict]:
    """
    One debate round for a single student agent.

    Each example prompt includes:
      - The agent's own previous answer
      - Both peers' previous answers
      - The teacher's personalised guidance for this agent

    Returns a list of per-example dicts:
      { "output": str, "reasoning": str, "raw_text": str }
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

        # This agent's own answer from the previous round
        my_output    = prev_round_preds[label][idx]["output"]
        my_reasoning = prev_round_preds[label][idx]["reasoning"]

        # Both peers' answers from the previous round
        peer_responses: List[Tuple[str, str, str]] = []
        for peer_label, peer_name in PEER_LABELS[label]:
            peer_output    = prev_round_preds[peer_label][idx]["output"]
            peer_reasoning = prev_round_preds[peer_label][idx]["reasoning"]
            peer_responses.append((peer_label, peer_output, peer_reasoning))

        # Teacher's personalised guidance for this agent
        t_guidance = teacher_guidance[idx].get(f"guidance_{label}", "")

        prompt = build_debate_prompt(
            code, inp,
            my_label=label,
            round_num=round_num,
            my_output=my_output,
            my_reasoning=my_reasoning,
            peer_responses=peer_responses,
            teacher_guidance=t_guidance,
        )

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj  = _first_json_object(raw)
        out  = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        expl = str(obj.get("explanation", "") or "") if obj else ""

        # Fallback: retain previous answer if model returns nothing useful
        if not out.strip():
            out  = my_output
            expl = f"[round-{round_num} fallback] {my_reasoning}"

        results.append({"output": out, "reasoning": expl, "raw_text": raw})
        print(f"-> revised: {out[:60]!r}")

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
    all_preds:    List[Dict[str, List[Dict]]],   # index 0=Phase0, 1..5=rounds
    all_corr:     List[Dict[str, List[str]]],    # parallel correctness
    all_guidance: List[Optional[List[Dict[str, str]]]],  # teacher guidance per round (None for Phase 0)
    json_path: str = DEFAULT_RESULTS_JSON,
    csv_path:  str = DEFAULT_RESULTS_CSV,
) -> None:
    """
    Writes a single JSON + CSV with all phases side-by-side.

    Column naming:
      Phase 0 (baseline) : {lbl}_output,    {lbl}_reasoning,    {lbl}_Correctness
      Debate round N     : {lbl}_output_dN, {lbl}_reasoning_dN, {lbl}_Correctness_dN

    Teacher guidance is stored in JSON only (not CSV) to keep the CSV manageable:
      teacher_guidance_dN: { "guidance_M": ..., "guidance_P": ..., "guidance_Q": ... }
    """
    labels = ["M", "P", "Q"]
    rows   = []

    for idx, ex in enumerate(dataset):
        row = {
            "code":        ex.get("code",   ""),
            "input":       ex.get("input",  ""),
            "gold_output": ex.get("output", ""),
        }

        # Student predictions + correctness for every phase
        for phase_idx, (preds, corr) in enumerate(zip(all_preds, all_corr)):
            suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
            for lbl in labels:
                p = preds.get(lbl, [])
                c = corr.get(lbl,  [])
                row[f"{lbl}_output{suffix}"]      = p[idx]["output"]    if idx < len(p) else ""
                row[f"{lbl}_reasoning{suffix}"]   = p[idx]["reasoning"] if idx < len(p) else ""
                row[f"{lbl}_Correctness{suffix}"] = c[idx]              if idx < len(c) else ""

        # Teacher guidance (JSON only, stored as nested dict per round)
        for round_idx, guidance_list in enumerate(all_guidance):
            if guidance_list is None:
                continue   # Phase 0 has no teacher guidance
            rn = round_idx  # guidance_list[0] = guidance for round 1, etc.
            row[f"teacher_guidance_d{rn}"] = {
                "guidance_M": guidance_list[idx].get("guidance_M", ""),
                "guidance_P": guidance_list[idx].get("guidance_P", ""),
                "guidance_Q": guidance_list[idx].get("guidance_Q", ""),
            }

        rows.append(row)

    # ---- JSON ---------------------------------------------------------------
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved -> {json_path}")

    # ---- CSV  (student columns only; teacher guidance too nested for flat CSV) ----
    fieldnames = ["code", "input", "gold_output"]
    for phase_idx in range(len(all_preds)):
        suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
        for lbl in labels:
            fieldnames += [
                f"{lbl}_output{suffix}",
                f"{lbl}_reasoning{suffix}",
                f"{lbl}_Correctness{suffix}",
            ]

    # Flatten teacher guidance into CSV-friendly columns
    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        for lbl in labels:
            fieldnames.append(f"teacher_guidance_{lbl}_d{round_num}")

    # Rebuild rows with flat teacher guidance for CSV
    flat_rows = []
    for idx, row in enumerate(rows):
        flat_row = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
            nested = row.get(f"teacher_guidance_d{round_num}", {})
            for lbl in labels:
                flat_row[f"teacher_guidance_{lbl}_d{round_num}"] = nested.get(f"guidance_{lbl}", "")
        flat_rows.append(flat_row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"CSV results saved  -> {csv_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(
    all_corr: List[Dict[str, List[str]]],
    labels: List[str] = ("M", "P", "Q"),
) -> None:
    num_phases  = len(all_corr)
    phase_names = ["Phase-0 (base)"] + [f"Round-{r}" for r in range(1, num_phases)]
    col_w       = 22

    sep = "=" * (8 + col_w * num_phases)
    print("\n" + sep)
    print("SUMMARY  (teacher-guided ToM debate)")
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

    print(f"Dataset        : {dataset_path}")
    print(f"Debate rounds  : {NUM_DEBATE_ROUNDS}")
    print(f"Teacher model  : {MODEL_TEACHER}")
    dataset = load_dataset(dataset_path)
    print(f"Loaded         : {len(dataset)} examples\n")

    # Accumulators for all phases
    all_preds:    List[Dict[str, List[Dict]]]            = []
    all_corr:     List[Dict[str, List[str]]]             = []
    # all_guidance[0] = None (Phase 0 has no teacher)
    # all_guidance[i] = teacher guidance list for debate round i (i >= 1)
    all_guidance: List[Optional[List[Dict[str, str]]]]   = []

    # -----------------------------------------------------------------------
    # PHASE 0 - Baseline: each agent predicts independently (no teacher)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# PHASE 0 - BASELINE (independent predictions)")
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
    all_guidance.append(None)       # no teacher in Phase 0

    # -----------------------------------------------------------------------
    # ROUNDS 1 - NUM_DEBATE_ROUNDS
    #
    # Each round:
    #   A) Teacher reads all agents' PREVIOUS answers -> per-agent guidance
    #   B) Each agent revises its answer using teacher guidance + peer answers
    #   C) Judge scores revised answers
    #   D) Advance prev_preds chain for the next round
    # -----------------------------------------------------------------------
    prev_preds = phase0_preds

    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        print("\n" + "#" * 60)
        print(f"# DEBATE ROUND {round_num} / {NUM_DEBATE_ROUNDS}")
        print("#" * 60)

        # --- Step A: Teacher generates personalised guidance ----------------
        teacher_guidance = run_teacher(
            dataset,
            prev_round_preds=prev_preds,
            round_num=round_num,
            max_new_tokens=512,
        )

        # --- Step B: Each agent debates with teacher guidance ---------------
        round_preds: Dict[str, List[Dict]] = {}
        for label, model_name in CONTESTANT_MODELS:
            round_preds[label] = run_debate_round(
                label, model_name, dataset,
                prev_round_preds=prev_preds,
                teacher_guidance=teacher_guidance,
                round_num=round_num,
                max_new_tokens=max_new_tokens,
            )

        # --- Step C: Judge scores this round --------------------------------
        round_corr = run_judge(
            dataset, round_preds,
            phase_label=f"Debate Round {round_num}",
            max_new_tokens=128,
        )

        all_preds.append(round_preds)
        all_corr.append(round_corr)
        all_guidance.append(teacher_guidance)

        # --- Step D: Advance chain for next round ---------------------------
        prev_preds = round_preds

    # -----------------------------------------------------------------------
    # Save all phases + teacher guidance to JSON + CSV
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# SAVING RESULTS (all phases + teacher guidance)")
    print("#" * 60)

    write_outputs(
        dataset,
        all_preds,
        all_corr,
        all_guidance,
        json_path=DEFAULT_RESULTS_JSON,
        csv_path=DEFAULT_RESULTS_CSV,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_summary(all_corr)


if __name__ == "__main__":
    main()