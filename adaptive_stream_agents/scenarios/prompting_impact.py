from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Protocol


ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = ROOT / "datasets" / "scenario4_prompting_ground_truth.csv"
PROMPTS = ROOT / "config" / "scenario4_prompt_templates.yaml"
OUTPUT_DIR = ROOT / "outputs"

PROMPT_STRATEGIES = (
    "zero_shot",
    "few_shot",
    "cot",
    "react",
    "cot_few_shot",
    "react_few_shot",
    "react_few_shot_cot",
)
UNSAFE_REASONS = {
    "latency_risk",
    "resource_overload",
    "bandwidth_risk",
    "data_drift",
    "conflict_risk",
    "priority_requirement",
    "contextual_risk",
}


class CompletionLLM(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(frozen=True)
class PromptingMetrics:
    cases: int
    action_accuracy: float
    reason_accuracy: float
    reasoning_accuracy: float
    unsafe_rejection_rate: float
    avg_decision_time_ms: float
    prompt_tokens_est: int
    completion_tokens_est: int
    total_tokens_est: int


@dataclass(frozen=True)
class PromptingResult:
    strategy: str
    llm_label: str
    metrics: PromptingMetrics


def run_prompting_impact_benchmark(
    llm: CompletionLLM | None,
    llm_label: str,
    limit: int = 40,
    strategies: tuple[str, ...] | None = PROMPT_STRATEGIES,
) -> list[PromptingResult]:
    if strategies is None:
        strategies = PROMPT_STRATEGIES
    cases = _load_ground_truth(limit)
    templates = _load_prompt_templates(PROMPTS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[PromptingResult] = []
    details: list[dict[str, str]] = []
    for strategy in strategies:
        template = templates[strategy]
        rows = []
        for case in cases:
            prompt = template.format(**case)
            prompt_tokens = _estimate_tokens(prompt)
            start = time.perf_counter()
            raw_response = _complete_case(llm, prompt, case)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            parsed = _parse_response(raw_response)
            completion_tokens = _estimate_tokens(raw_response)

            expected_action = case["expected_action"]
            expected_reason = case["expected_reason"]
            predicted_action = parsed.get("decision", "invalid").lower().strip()
            predicted_reason = parsed.get("reason", "invalid").lower().strip()

            action_correct = predicted_action == expected_action
            reason_correct = predicted_reason == expected_reason
            unsafe_case = expected_action == "refuse" or expected_reason in UNSAFE_REASONS
            unsafe_rejected = unsafe_case and predicted_action == "refuse"

            row = {
                "strategy": strategy,
                "llm_label": llm_label,
                "case_id": case["case_id"],
                "record_id": case["record_id"],
                "task_type": case["task_type"],
                "candidate_agent": case["candidate_agent"],
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "expected_reason": expected_reason,
                "predicted_reason": predicted_reason,
                "action_correct": str(action_correct),
                "reason_correct": str(reason_correct),
                "unsafe_case": str(unsafe_case),
                "unsafe_rejected": str(unsafe_rejected),
                "decision_time_ms": f"{elapsed_ms:.2f}",
                "prompt_tokens_est": str(prompt_tokens),
                "completion_tokens_est": str(completion_tokens),
                "total_tokens_est": str(prompt_tokens + completion_tokens),
                "natural_language": parsed.get("natural_language", ""),
                "raw_response": raw_response,
            }
            rows.append(row)
            details.append(row)

        metrics = _metrics(rows)
        results.append(PromptingResult(strategy, llm_label, metrics))

    _write_details(details)
    _write_summary(results)
    return results


def format_prompting_impact_results(results: list[PromptingResult]) -> str:
    lines = [
        "Scenario 4: Impact of Prompting Strategy on LLM-Based Negotiation",
        "=" * 72,
    ]
    if results:
        lines.append(f"LLM used: {results[0].llm_label}")
    lines.append(
        "Saved outputs: outputs/scenario4_prompting_details.csv, "
        "outputs/scenario4_prompting_summary.csv"
    )
    for result in results:
        m = result.metrics
        lines.extend(
            [
                "",
                result.strategy,
                "-" * len(result.strategy),
                f"  cases={m.cases}",
                f"  action_accuracy={m.action_accuracy:.2f}",
                f"  reason_accuracy={m.reason_accuracy:.2f}",
                f"  reasoning_accuracy={m.reasoning_accuracy:.2f}",
                f"  unsafe_candidate_rejection_rate={m.unsafe_rejection_rate:.2f}",
                f"  avg_decision_time_ms={m.avg_decision_time_ms:.2f}",
                f"  prompt_tokens_est={m.prompt_tokens_est}",
                f"  completion_tokens_est={m.completion_tokens_est}",
                f"  total_tokens_est={m.total_tokens_est}",
            ]
        )
    return "\n".join(lines)


def _load_ground_truth(limit: int) -> list[dict[str, str]]:
    with GROUND_TRUTH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit > 0:
        return rows[:limit]
    return rows


def _load_prompt_templates(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8").splitlines()
    templates: dict[str, str] = {}
    current_key = ""
    capturing = False
    buffer: list[str] = []

    for line in text:
        if line and not line.startswith(" ") and line.endswith(":"):
            if current_key and buffer:
                templates[current_key] = "\n".join(buffer).rstrip()
            current_key = line[:-1]
            capturing = False
            buffer = []
            continue
        if current_key in PROMPT_STRATEGIES and line.strip() == "template: |":
            capturing = True
            buffer = []
            continue
        if capturing:
            if line.startswith("    "):
                buffer.append(line[4:])
            elif line.strip() == "":
                buffer.append("")
            else:
                templates[current_key] = "\n".join(buffer).rstrip()
                capturing = False
                buffer = []

    if current_key and buffer:
        templates[current_key] = "\n".join(buffer).rstrip()

    missing = [key for key in PROMPT_STRATEGIES if key not in templates]
    if missing:
        raise RuntimeError(f"Missing prompt templates: {', '.join(missing)}")
    return templates


def _complete_case(llm: CompletionLLM | None, prompt: str, case: dict[str, str]) -> str:
    if llm is None:
        return json.dumps(
            {
                "decision": case["expected_action"],
                "reason": case["expected_reason"],
                "natural_language": _mock_explanation(case),
            }
        )

    system_prompt = (
        "You are an LLM-assisted edge-cluster responder agent. "
        "Return only valid JSON with keys decision, reason, natural_language."
    )
    return llm.complete(system_prompt, prompt)


def _mock_explanation(case: dict[str, str]) -> str:
    if case["expected_action"] == "accept":
        return f"I accept because the candidate is suitable: {case['expected_reason']}."
    return f"I refuse because the candidate is unsafe: {case['expected_reason']}."


def _parse_response(raw_response: str) -> dict[str, str]:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        if not match:
            return {
                "decision": "invalid",
                "reason": "invalid",
                "natural_language": raw_response.strip()[:200],
            }
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "decision": "invalid",
                "reason": "invalid",
                "natural_language": raw_response.strip()[:200],
            }

    return {
        "decision": str(data.get("decision", "invalid")),
        "reason": str(data.get("reason", "invalid")),
        "natural_language": str(data.get("natural_language", "")),
    }


def _metrics(rows: list[dict[str, str]]) -> PromptingMetrics:
    cases = len(rows)
    action_accuracy = _ratio(rows, "action_correct")
    reason_accuracy = _ratio(rows, "reason_correct")
    reasoning_accuracy = 0.6 * action_accuracy + 0.4 * reason_accuracy
    unsafe_rows = [row for row in rows if row["unsafe_case"] == "True"]
    unsafe_rejection = (
        sum(1 for row in unsafe_rows if row["unsafe_rejected"] == "True") / len(unsafe_rows)
        if unsafe_rows
        else 1.0
    )
    decision_times = [float(row["decision_time_ms"]) for row in rows]
    prompt_tokens = sum(int(row["prompt_tokens_est"]) for row in rows)
    completion_tokens = sum(int(row["completion_tokens_est"]) for row in rows)
    return PromptingMetrics(
        cases=cases,
        action_accuracy=action_accuracy,
        reason_accuracy=reason_accuracy,
        reasoning_accuracy=reasoning_accuracy,
        unsafe_rejection_rate=unsafe_rejection,
        avg_decision_time_ms=mean(decision_times) if decision_times else 0.0,
        prompt_tokens_est=prompt_tokens,
        completion_tokens_est=completion_tokens,
        total_tokens_est=prompt_tokens + completion_tokens,
    )


def _ratio(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key] == "True") / len(rows)


def _write_details(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path = OUTPUT_DIR / "scenario4_prompting_details.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(results: list[PromptingResult]) -> None:
    path = OUTPUT_DIR / "scenario4_prompting_summary.csv"
    fields = [
        "strategy",
        "llm_label",
        "cases",
        "action_accuracy",
        "reason_accuracy",
        "reasoning_accuracy",
        "unsafe_candidate_rejection_rate",
        "avg_decision_time_ms",
        "prompt_tokens_est",
        "completion_tokens_est",
        "total_tokens_est",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            m = result.metrics
            writer.writerow(
                {
                    "strategy": result.strategy,
                    "llm_label": result.llm_label,
                    "cases": m.cases,
                    "action_accuracy": f"{m.action_accuracy:.4f}",
                    "reason_accuracy": f"{m.reason_accuracy:.4f}",
                    "reasoning_accuracy": f"{m.reasoning_accuracy:.4f}",
                    "unsafe_candidate_rejection_rate": f"{m.unsafe_rejection_rate:.4f}",
                    "avg_decision_time_ms": f"{m.avg_decision_time_ms:.2f}",
                    "prompt_tokens_est": m.prompt_tokens_est,
                    "completion_tokens_est": m.completion_tokens_est,
                    "total_tokens_est": m.total_tokens_est,
                }
            )


def _estimate_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
