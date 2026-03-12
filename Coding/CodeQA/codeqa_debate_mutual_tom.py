# =============================================================================
# CodeQA Multi-Model Evaluator  -  Mutual Theory-of-Mind Debate
#
# Dataset : CodeQA  (each example has: code, question, answer)
#           The task is to answer a natural-language question about a code
#           snippet (e.g. "What does the code do?", "Does it raise an error?")
#
# Models:
#   M       = Mistral-7B-Instruct-v0.3         (student agent)
#   P       = Phi-4-mini-instruct               (student agent)
#   Q       = Qwen2.5-7B-Instruct               (student agent)
#   Teacher = Qwen3-Coder-30B-A3B-Instruct      (teacher / ToM guide, sees gold answer privately)
#   Judge   = Qwen2.5-7B-Instruct               (evaluator, unchanged)
#
# Research objective: Mutual Theory-of-Mind (MToM)
# -------------------------------------------------
# At Phase 0 (baseline) each agent produces:
#   - predicted_output  : the answer
#   - explanation       : reasoning
#   - confidence_score  : float 0.0-1.0  (INTERNAL ONLY — never shown to peers)
#
# The confidence_score is a backend self-calibration signal.  It is:
#   (a) used by the teacher to gauge how certain each agent is, so it can
#       tailor guidance more precisely (e.g. "you are wrong AND confident —
#       re-examine assumption X" vs "you are wrong but uncertain — here is
#       a nudge")
#   (b) stored in JSON/CSV for offline analysis
#   (c) NEVER forwarded to peer agents, preserving a clean mutual-ToM
#       scenario where each agent reasons about peers' minds purely from
#       their stated answers and reasoning.
#
# Pipeline
# --------
# Phase 0      : Each student independently answers + self-rates confidence
#                --> Judge scores Phase 0
#
# Rounds 1-5   : Mixed schedule controlled by TEACHER_INTERVAL = 2
#
#   Pure-debate rounds  (round_num % TEACHER_INTERVAL != 0):
#     Step A  : Each student sees its own previous answer + both peers'
#               previous answers (NO confidence scores shown) and revises.
#     Step B  : Judge scores revised answers.
#
#   Teacher rounds  (round_num % TEACHER_INTERVAL == 0):
#     Step A  : Teacher reads ALL three students' previous answers AND their
#               (private) confidence scores, then produces targeted MToM
#               guidance per agent — confidence informs the guidance tone
#               without being revealed to peers.
#     Step B  : Each student receives peer answers + teacher's personal
#               guidance and revises its answer + confidence score.
#     Step C  : Judge scores revised answers.
#
#   With TEACHER_INTERVAL=2 and NUM_DEBATE_ROUNDS=5 the schedule is:
#     Round 1 -> pure debate
#     Round 2 -> teacher intervenes
#     Round 3 -> pure debate
#     Round 4 -> teacher intervenes
#     Round 5 -> pure debate
#
# Output    : Single JSON + CSV with ALL rounds side-by-side.
#
# CSV columns
# -----------
# code, question, gold_answer,
# M_output, M_confidence, M_reasoning, M_Correctness,         <- Phase 0
# P_output, P_confidence, P_reasoning, P_Correctness,
# Q_output, Q_confidence, Q_reasoning, Q_Correctness,
# M_output_d1, M_confidence_d1, M_reasoning_d1, M_Correctness_d1,  <- Round 1
# ...repeated for d2, d3, d4, d5...
# teacher_guidance_M_d2, teacher_guidance_P_d2, teacher_guidance_Q_d2, <- teacher rounds
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

NUM_DEBATE_ROUNDS      = 5

# Teacher fires every TEACHER_INTERVAL rounds (e.g. 2 = rounds 2, 4, ...).
# All other rounds are pure peer-debate with no teacher.
TEACHER_INTERVAL       = 2

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


def _parse_confidence(obj: Optional[dict], fallback: float = 0.5) -> float:
    """Safely extract confidence_score from parsed JSON, clamped to [0, 1]."""
    if not obj:
        return fallback
    raw = obj.get("confidence_score", fallback)
    try:
        val = float(raw)
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_contestant_prompt(code: str, question: str) -> str:
    """
    Phase 0: independent QA.
    The agent produces an answer, a confidence score (backend only), and
    an explanation.  The confidence score is used internally by the teacher
    for Mutual Theory-of-Mind calibration — it is never shared with peers.
    """
    return f"""You are given a Python code snippet and a question about it.

Task:
Read the code carefully and answer the question. Return your answer, a self-assessed
confidence score, and a brief explanation of your reasoning.

Definition of confidence_score:
- A float between 0.0 and 1.0 representing how certain you are in your answer.
- 1.0 = completely certain, 0.5 = unsure, 0.0 = guessing.
- Be honest: an accurate confidence score is more useful than an inflated one.

Definition of explanation:
- A short, high-level description of your reasoning. How did you arrive at the answer?
- Make it concise — 2-3 sentences.

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


def build_teacher_prompt(
    code: str,
    question: str,
    gold_answer: str,
    round_num: int,
    agent_responses: List[Tuple[str, str, float, str]],  # [(label, answer, confidence, reasoning)]
) -> str:
    """
    Teacher (Mutual Theory-of-Mind) prompt for CodeQA.

    The teacher privately receives:
      - the gold answer
      - each agent's answer, confidence score, and reasoning

    The confidence score is a key MToM signal: a wrong-but-confident agent
    needs stronger redirection than a wrong-but-uncertain one.
    The teacher must NOT reveal the gold answer verbatim, and must NOT
    mention confidence scores in its guidance (they stay backend-only).

    Returns JSON:
    {
      "guidance_M": "...",
      "guidance_P": "...",
      "guidance_Q": "..."
    }
    """
    agent_block = ""
    for label, answer, confidence, reasoning in agent_responses:
        agent_block += f"""
--- Agent {label} ({AGENT_DISPLAY_NAME[label]}) ---
Answer     : {answer}
Confidence : {confidence:.2f}   [PRIVATE — do not mention this number in guidance]
Reasoning  : {reasoning}
"""

    return f"""You are an expert Python teacher and Mutual Theory-of-Mind reasoner participating in \
Round {round_num} of a {NUM_DEBATE_ROUNDS}-round multi-agent debate.

Three small language model agents (M, P, Q) are each trying to answer a question about a \
Python code snippet. You have received their latest answers, confidence scores, and reasoning.

[PRIVATE — FOR YOUR EYES ONLY]
The verified correct answer to the question is: "{gold_answer}"
Each agent's confidence score is also provided privately above.
Use BOTH pieces of information to calibrate your guidance:
  - An agent that is WRONG and CONFIDENT (score >= 0.7) needs a firm, specific correction
    that directly challenges their mistaken assumption.
  - An agent that is WRONG and UNCERTAIN (score < 0.7) benefits from a focused nudge
    pointing them toward the key part of the code they are misreading.
  - An agent that is CORRECT should be encouraged to hold their position.

IMPORTANT RULES:
  - Do NOT reveal the gold answer verbatim in your guidance.
  - Do NOT mention the confidence score numbers in your guidance text.
  - Do NOT tell agents what their peers' confidence scores are.
  - Guide each agent individually based on their own reasoning flaw.

Your role - Mutual Theory-of-Mind guidance:
- For EACH agent, compare their answer against the correct answer above.
- Identify the specific step or assumption in each agent's reasoning that is flawed
  (or confirm it is correct).
- If an agent is WRONG: Tell them clearly which part of the code or reasoning led
  them astray (e.g., "You misread line 3 — the loop condition is X not Y").
  Guide them firmly toward the correct answer without quoting it word-for-word.
- If an agent is CORRECT: Say "Your answer is on the right track.
  Maintain your position confidently even if peers disagree."

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

=== Question ===
{question}

=== Agent Answers (Round {round_num - 1} output) ===
{agent_block}
"""


def build_debate_prompt(
    code: str,
    question: str,
    my_label: str,
    round_num: int,
    my_output: str,
    my_reasoning: str,
    peer_responses: List[Tuple[str, str, str]],   # [(label, answer, reasoning)] — NO confidence
    teacher_guidance: str,                         # "" on pure-debate rounds
) -> str:
    """
    Student debate prompt for CodeQA.

    Peer responses intentionally omit confidence scores — agents reason about
    each other's minds purely through stated answers and explanations (MToM).
    """
    peer_block = ""
    for peer_label, peer_output, peer_reasoning in peer_responses:
        peer_block += f"""
--- Agent {peer_label} ({AGENT_DISPLAY_NAME[peer_label]}) ---
Answer: {peer_output}
Reasoning: {peer_reasoning}
"""

    if teacher_guidance.strip():
        teacher_section = f"""
=== Teacher Guidance (personalised for Agent {my_label}) ===
{teacher_guidance}
"""
        teacher_job_line = (
            "- Read the teacher's guidance carefully - it is targeted specifically at "
            "your reasoning.\n"
            "- Re-examine the code and the question, paying attention to the specific "
            "concern the teacher raised.\n"
        )
        closing = (
            "Now carefully re-analyse the code using the teacher's guidance and return "
            "your (possibly revised) answer as valid JSON."
        )
    else:
        teacher_section  = ""
        teacher_job_line = ""
        closing = (
            "Now carefully re-analyse the code and return your "
            "(possibly revised) answer as valid JSON."
        )

    return f"""You are Agent {my_label} ({AGENT_DISPLAY_NAME[my_label]}), participating in \
Round {round_num} of a {NUM_DEBATE_ROUNDS}-round multi-agent debate for code question answering.

You will receive:
1. The Python code snippet and the question about it.
2. YOUR answer and reasoning from the previous round (which may be correct or incorrect).
3. The answers and reasoning of the other two agents from the previous round.

Your job:
{teacher_job_line}- Consider the other agents' reasoning critically - they may be right or wrong.
- Do NOT blindly follow the majority. A lone agent can be right while two are wrong.
- If another agent's reasoning exposes a flaw in your answer, revise it.
- If you remain confident in your answer, keep it and clearly explain why.

Rules for predicted_output (your answer):
- Give a direct, concise answer to the question (e.g. "Yes", "No", "a suite", "an error").
- Match the expected answer style: short noun phrases or Yes/No where appropriate.
- Do not include the question or the code in your answer.

Also update your confidence_score to reflect your current certainty after seeing all answers.

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
Reasoning: {my_reasoning}

=== Other Agents' Answers from Previous Round ===
{peer_block}{teacher_section}
{closing}
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
                    "input":  row.get("question", row.get("input",  "")),
                    "output": row.get("answer",   row.get("output", row.get("gold_answer", ""))),
                })
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of examples.")

    normalised = []
    for i, ex in enumerate(data):
        normalised.append({
            "id":     ex.get("id", f"idx_{i}"),
            "code":   ex.get("code",  ""),
            "input":  ex.get("question", ex.get("input",  "")),
            "output": ex.get("answer",   ex.get("output", "")),
        })
    return normalised


# ---------------------------------------------------------------------------
# Phase 0 - baseline: each agent independently predicts + self-rates confidence
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

    confidence is stored internally and passed to the teacher.
    It is NOT forwarded to peer agents at any point.
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

        obj        = _first_json_object(raw)
        out        = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        confidence = _parse_confidence(obj)
        expl       = str(obj.get("explanation", "") or "") if obj else ""

        results.append({
            "output":     out,
            "confidence": confidence,
            "reasoning":  expl,
            "raw_text":   raw,
        })
        print(f"-> predicted: {out[:60]!r}  conf={confidence:.2f}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Teacher pass - generates per-agent MToM guidance using private confidence
# ---------------------------------------------------------------------------
def run_teacher(
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    round_num: int,
    max_new_tokens: int = 512,
) -> List[Dict[str, str]]:
    """
    For each example, the teacher privately receives:
      - the gold answer
      - each agent's answer, confidence score (MToM signal), and reasoning

    It produces personalised guidance per agent.  Confidence scores inform
    the TONE of guidance (firm vs gentle) but are never written into the
    guidance text — peers never learn each other's confidence values.

    Returns a list (one entry per example) of dicts:
      {
        "guidance_M": str,
        "guidance_P": str,
        "guidance_Q": str,
        "raw_text":   str,
      }
    """
    print(f"\n{'='*60}")
    print(f"TEACHER PASS  Round {round_num}  model={MODEL_TEACHER}")
    print(f"{'='*60}")

    runner  = ModelRunner(MODEL_TEACHER)
    results = []

    for idx, ex in enumerate(dataset):
        code        = ex.get("code",   "")
        inp         = ex.get("input",  "")
        gold_answer = ex.get("output", "")
        ex_id       = ex.get("id",     f"idx_{idx}")

        # Collect all three agents' latest answers + confidence (private MToM signal)
        agent_responses: List[Tuple[str, str, float, str]] = []
        for label, _ in CONTESTANT_MODELS:
            out        = prev_round_preds[label][idx]["output"]
            confidence = prev_round_preds[label][idx].get("confidence", 0.5)
            expl       = prev_round_preds[label][idx]["reasoning"]
            agent_responses.append((label, out, confidence, expl))

        prompt = build_teacher_prompt(code, inp, gold_answer, round_num, agent_responses)

        print(f"  [Teacher] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj = _first_json_object(raw)

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
# Debate round - each agent revises its answer (peers never see confidence)
# ---------------------------------------------------------------------------
def run_debate_round(
    label: str,
    model_name: str,
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    teacher_guidance: Optional[List[Dict[str, str]]],  # None on pure-debate rounds
    round_num: int,
    max_new_tokens: int,
) -> List[Dict]:
    """
    One debate round for a single student agent.

    Peer answers and reasoning are shared; peer confidence scores are NOT.
    This preserves the Mutual Theory-of-Mind design: agents must infer
    each other's certainty from reasoning quality, not from a raw number.

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

        my_output    = prev_round_preds[label][idx]["output"]
        my_reasoning = prev_round_preds[label][idx]["reasoning"]

        # Peers: share answer + reasoning only — confidence stays hidden
        peer_responses: List[Tuple[str, str, str]] = []
        for peer_label, peer_name in PEER_LABELS[label]:
            peer_output    = prev_round_preds[peer_label][idx]["output"]
            peer_reasoning = prev_round_preds[peer_label][idx]["reasoning"]
            peer_responses.append((peer_label, peer_output, peer_reasoning))

        t_guidance = (
            teacher_guidance[idx].get(f"guidance_{label}", "")
            if teacher_guidance is not None else ""
        )

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

        obj        = _first_json_object(raw)
        out        = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        confidence = _parse_confidence(obj, fallback=prev_round_preds[label][idx].get("confidence", 0.5))
        expl       = str(obj.get("explanation", "") or "") if obj else ""

        if not out.strip():
            out        = my_output
            confidence = prev_round_preds[label][idx].get("confidence", 0.5)
            expl       = f"[round-{round_num} fallback] {my_reasoning}"

        results.append({
            "output":     out,
            "confidence": confidence,
            "reasoning":  expl,
            "raw_text":   raw,
        })
        print(f"-> revised: {out[:60]!r}  conf={confidence:.2f}")

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
                verdict = "INCORRECT"

            correctness[label].append(verdict)
            print(f"    [{label}] pred={pred[:40]!r}  ->  {verdict}")

    runner.unload()
    return correctness


# ---------------------------------------------------------------------------
# Output writers - all phases in one JSON + CSV
# ---------------------------------------------------------------------------
def write_outputs(
    dataset: List[Dict],
    all_preds:    List[Dict[str, List[Dict]]],
    all_corr:     List[Dict[str, List[str]]],
    all_guidance: List[Optional[List[Dict[str, str]]]],
    json_path: str = DEFAULT_RESULTS_JSON,
    csv_path:  str = DEFAULT_RESULTS_CSV,
) -> None:
    """
    Writes a single JSON + CSV with all phases side-by-side.

    confidence_score is stored in both JSON and CSV for analysis.
    It appears alongside each agent's answer but is clearly labelled
    as a backend-only signal in the column header.

    Column naming:
      Phase 0 (baseline) : {lbl}_output, {lbl}_confidence, {lbl}_reasoning, {lbl}_Correctness
      Debate round N     : {lbl}_output_dN, {lbl}_confidence_dN, {lbl}_reasoning_dN, {lbl}_Correctness_dN

    Teacher guidance is stored in JSON and in flat CSV columns.
    """
    labels = ["M", "P", "Q"]
    rows   = []

    for idx, ex in enumerate(dataset):
        row = {
            "code":        ex.get("code",   ""),
            "question":    ex.get("input",  ""),
            "gold_answer": ex.get("output", ""),
        }

        for phase_idx, (preds, corr) in enumerate(zip(all_preds, all_corr)):
            suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
            for lbl in labels:
                p = preds.get(lbl, [])
                c = corr.get(lbl,  [])
                row[f"{lbl}_output{suffix}"]      = p[idx]["output"]              if idx < len(p) else ""
                row[f"{lbl}_confidence{suffix}"]  = p[idx].get("confidence", "")  if idx < len(p) else ""
                row[f"{lbl}_reasoning{suffix}"]   = p[idx]["reasoning"]           if idx < len(p) else ""
                row[f"{lbl}_Correctness{suffix}"] = c[idx]                        if idx < len(c) else ""

        for round_idx, guidance_list in enumerate(all_guidance):
            if guidance_list is None:
                continue
            rn = round_idx
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

    # ---- CSV ----------------------------------------------------------------
    fieldnames = ["code", "question", "gold_answer"]
    for phase_idx in range(len(all_preds)):
        suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
        for lbl in labels:
            fieldnames += [
                f"{lbl}_output{suffix}",
                f"{lbl}_confidence{suffix}",   # backend MToM signal
                f"{lbl}_reasoning{suffix}",
                f"{lbl}_Correctness{suffix}",
            ]

    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        for lbl in labels:
            fieldnames.append(f"teacher_guidance_{lbl}_d{round_num}")

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
    def _round_label(r):
        if r == 0:
            return "Phase-0 (base)"
        return f"Round-{r}[T]" if r % TEACHER_INTERVAL == 0 else f"Round-{r}[D]"
    phase_names = [_round_label(r) for r in range(num_phases)]
    col_w       = 22

    sep = "=" * (8 + col_w * num_phases)
    print("\n" + sep)
    print("SUMMARY  (mutual ToM debate  -  CodeQA)")
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
    print(f"Debate rounds  : {NUM_DEBATE_ROUNDS}")
    print(f"Teacher model  : {MODEL_TEACHER}")
    print(f"MToM mode      : confidence_score is backend-only (not shared with peers)")
    dataset = load_dataset(dataset_path)
    print(f"Loaded         : {len(dataset)} examples\n")

    all_preds:    List[Dict[str, List[Dict]]]            = []
    all_corr:     List[Dict[str, List[str]]]             = []
    all_guidance: List[Optional[List[Dict[str, str]]]]   = []

    # -----------------------------------------------------------------------
    # PHASE 0 - Baseline: each agent answers + self-rates confidence
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# PHASE 0 - BASELINE (independent answers + private confidence)")
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
    all_guidance.append(None)

    # -----------------------------------------------------------------------
    # ROUNDS 1 - NUM_DEBATE_ROUNDS  (mixed schedule)
    # -----------------------------------------------------------------------
    prev_preds = phase0_preds

    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        is_teacher_round = (round_num % TEACHER_INTERVAL == 0)
        round_type       = "TEACHER-GUIDED (MToM)" if is_teacher_round else "PURE DEBATE"

        print("\n" + "#" * 60)
        print(f"# DEBATE ROUND {round_num} / {NUM_DEBATE_ROUNDS}  [{round_type}]")
        print("#" * 60)

        if is_teacher_round:
            # Teacher receives confidence scores privately for MToM calibration
            teacher_guidance = run_teacher(
                dataset,
                prev_round_preds=prev_preds,
                round_num=round_num,
                max_new_tokens=512,
            )
        else:
            teacher_guidance = None

        # Agents debate — peers see answers+reasoning only, never confidence
        round_preds: Dict[str, List[Dict]] = {}
        for label, model_name in CONTESTANT_MODELS:
            round_preds[label] = run_debate_round(
                label, model_name, dataset,
                prev_round_preds=prev_preds,
                teacher_guidance=teacher_guidance,
                round_num=round_num,
                max_new_tokens=max_new_tokens,
            )

        round_corr = run_judge(
            dataset, round_preds,
            phase_label=f"Round {round_num} ({round_type})",
            max_new_tokens=128,
        )

        all_preds.append(round_preds)
        all_corr.append(round_corr)
        all_guidance.append(teacher_guidance)

        prev_preds = round_preds

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# SAVING RESULTS (all phases + MToM confidence + teacher guidance)")
    print("#" * 60)

    write_outputs(
        dataset,
        all_preds,
        all_corr,
        all_guidance,
        json_path=DEFAULT_RESULTS_JSON,
        csv_path=DEFAULT_RESULTS_CSV,
    )

    print_summary(all_corr)


if __name__ == "__main__":
    main()
