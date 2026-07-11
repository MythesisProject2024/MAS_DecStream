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
INITIATOR_GROUND_TRUTH = ROOT / "datasets" / "scenario4_initiator_ground_truth.csv"
RESPONDER_GROUND_TRUTH = ROOT / "datasets" / "scenario4_prompting_ground_truth.csv"
PROMPTS = ROOT / "config" / "scenario4_initiator_prompt_templates.yaml"
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


class CompletionLLM(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(frozen=True)
class InitiatorPromptingMetrics:
    cases: int
    cfp_intent_accuracy: float
    cfp_keyword_coverage: float
    cfp_accuracy: float
    offloading_accuracy: float
    responder_action_accuracy: float
    avg_negotiation_rounds: float
    avg_decision_time_ms: float
    prompt_tokens_est: int
    completion_tokens_est: int
    total_tokens_est: int


@dataclass(frozen=True)
class InitiatorPromptingResult:
    strategy: str
    llm_label: str
    metrics: InitiatorPromptingMetrics


def run_initiator_prompting_benchmark(
    llm: CompletionLLM | None,
    llm_label: str,
    limit: int = 25,
    strategies: tuple[str, ...] | None = PROMPT_STRATEGIES,
    max_rounds: int = 2,
    consensus_margin: float = 0.15,
    epsilon: float = 0.03,
) -> list[InitiatorPromptingResult]:
    if strategies is None:
        strategies = PROMPT_STRATEGIES
    cases = _load_initiator_ground_truth(limit)
    responder_cases = _load_responder_ground_truth()
    templates = _load_prompt_templates(PROMPTS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[InitiatorPromptingResult] = []
    details: list[dict[str, str]] = []
    for strategy in strategies:
        template = templates[strategy]
        rows = []
        print(
            f"[Scenario 4-B] Running strategy {strategy} on {len(cases)} cases "
            f"with max_rounds={max_rounds}...",
            flush=True,
        )
        for case_index, case in enumerate(cases, start=1):
            print(
                f"[Scenario 4-B] {strategy}: case {case_index}/{len(cases)} "
                f"({case['case_id']})",
                flush=True,
            )
            prompt = template.format(**case)
            prompt_tokens = _estimate_tokens(prompt)
            start = time.perf_counter()
            raw_cfp = _complete_initiator(llm, prompt, case)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            completion_tokens = _estimate_tokens(raw_cfp)
            cfp = _parse_initiator_response(raw_cfp)

            expected_intent = case["expected_cfp_intent"]
            predicted_intent = cfp.get("cfp_intent", "invalid").lower().strip()
            keyword_coverage = _keyword_coverage(
                cfp.get("cfp_keywords", []),
                case["expected_cfp_keywords"],
            )
            cfp_intent_correct = predicted_intent == expected_intent
            cfp_accuracy = 0.7 * float(cfp_intent_correct) + 0.3 * keyword_coverage

            candidate_rows = responder_cases.get(case["record_id"], [])
            selected_host = "NONE"
            round_count = 0
            responder_rows: list[dict[str, str]] = []
            latest_valid_accepts: list[dict[str, str]] = []
            candidate_pool = list(candidate_rows)
            previous_best_utility: float | None = None

            for round_id in range(1, max_rounds + 1):
                round_count = round_id
                latest_valid_accepts = []
                for candidate in candidate_pool:
                    responder_prompt = _build_responder_prompt(
                        strategy=strategy,
                        case=case,
                        candidate=candidate,
                        cfp=cfp,
                        round_id=round_id,
                    )
                    prompt_tokens += _estimate_tokens(responder_prompt)
                    start = time.perf_counter()
                    raw_response = _complete_responder(llm, responder_prompt, candidate)
                    elapsed_ms += (time.perf_counter() - start) * 1000.0
                    completion_tokens += _estimate_tokens(raw_response)
                    parsed = _parse_responder_response(raw_response)

                    predicted_action = parsed.get("decision", "invalid")
                    expected_action = candidate["expected_action"]
                    action_correct = predicted_action == expected_action
                    hard_valid = expected_action == "accept"
                    if predicted_action == "accept" and hard_valid:
                        latest_valid_accepts.append(candidate)

                    responder_rows.append(
                        {
                            "round_id": str(round_id),
                            "candidate_agent": candidate["candidate_agent"],
                            "expected_action": expected_action,
                            "predicted_action": predicted_action,
                            "responder_action_correct": str(action_correct),
                            "responder_reason": parsed.get("reason", "invalid"),
                            "responder_message": parsed.get("natural_language", ""),
                        }
                    )

                if not latest_valid_accepts:
                    selected_host = "NONE"
                    break

                ranked_accepts = sorted(
                    latest_valid_accepts,
                    key=lambda row: float(row["utility"]),
                    reverse=True,
                )
                selected_host = ranked_accepts[0]["candidate_agent"]
                best_utility = float(ranked_accepts[0]["utility"])
                second_utility = float(ranked_accepts[1]["utility"]) if len(ranked_accepts) > 1 else None

                if _consensus_reached(
                    best_utility=best_utility,
                    second_utility=second_utility,
                    previous_best_utility=previous_best_utility,
                    consensus_margin=consensus_margin,
                    epsilon=epsilon,
                    round_id=round_id,
                ):
                    break

                previous_best_utility = best_utility
                candidate_pool = ranked_accepts

            expected_host = case["expected_final_host"]
            offloading_correct = selected_host == expected_host

            row = {
                "strategy": strategy,
                "llm_label": llm_label,
                "case_id": case["case_id"],
                "record_id": case["record_id"],
                "task_type": case["task_type"],
                "scenario_family": case["scenario_family"],
                "expected_cfp_intent": expected_intent,
                "predicted_cfp_intent": predicted_intent,
                "cfp_intent_correct": str(cfp_intent_correct),
                "cfp_keyword_coverage": f"{keyword_coverage:.4f}",
                "cfp_accuracy": f"{cfp_accuracy:.4f}",
                "expected_final_host": expected_host,
                "selected_host_after_negotiation": selected_host,
                "offloading_correct": str(offloading_correct),
                "negotiation_rounds": str(round_count),
                "responder_action_accuracy": f"{_responder_action_accuracy(responder_rows):.4f}",
                "decision_time_ms": f"{elapsed_ms:.2f}",
                "prompt_tokens_est": str(prompt_tokens),
                "completion_tokens_est": str(completion_tokens),
                "total_tokens_est": str(prompt_tokens + completion_tokens),
                "generated_cfp_message": cfp.get("cfp_message", ""),
                "raw_initiator_response": raw_cfp,
                "responder_transcript": json.dumps(responder_rows),
            }
            rows.append(row)
            details.append(row)

        metrics = _metrics(rows)
        results.append(InitiatorPromptingResult(strategy, llm_label, metrics))
        _write_details(details)
        _write_summary(results)

    return results


def format_initiator_prompting_results(results: list[InitiatorPromptingResult]) -> str:
    lines = [
        "Scenario 4-B: End-to-End LLM-MR-CNP Negotiation Evaluation",
        "=" * 64,
    ]
    if results:
        lines.append(f"LLM used by initiator and responders: {results[0].llm_label}")
    lines.append(
        "Saved outputs: outputs/scenario4b_negotiation_details.csv, "
        "outputs/scenario4b_negotiation_summary.csv"
    )
    for result in results:
        m = result.metrics
        lines.extend(
            [
                "",
                result.strategy,
                "-" * len(result.strategy),
                f"  cases={m.cases}",
                f"  cfp_intent_accuracy={m.cfp_intent_accuracy:.2f}",
                f"  cfp_keyword_coverage={m.cfp_keyword_coverage:.2f}",
                f"  cfp_accuracy={m.cfp_accuracy:.2f}",
                f"  offloading_accuracy={m.offloading_accuracy:.2f}",
                f"  responder_action_accuracy={m.responder_action_accuracy:.2f}",
                f"  avg_negotiation_rounds={m.avg_negotiation_rounds:.2f}",
                f"  avg_decision_time_ms={m.avg_decision_time_ms:.2f}",
                f"  prompt_tokens_est={m.prompt_tokens_est}",
                f"  completion_tokens_est={m.completion_tokens_est}",
                f"  total_tokens_est={m.total_tokens_est}",
            ]
        )
    return "\n".join(lines)


def _load_initiator_ground_truth(limit: int) -> list[dict[str, str]]:
    with INITIATOR_GROUND_TRUTH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit > 0 else rows


def _load_responder_ground_truth() -> dict[str, list[dict[str, str]]]:
    with RESPONDER_GROUND_TRUTH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["record_id"], []).append(row)
    return grouped


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
        raise RuntimeError(f"Missing initiator prompt templates: {', '.join(missing)}")
    return templates


def _complete_initiator(llm: CompletionLLM | None, prompt: str, case: dict[str, str]) -> str:
    if llm is None:
        return json.dumps(
            {
                "cfp_intent": case["expected_cfp_intent"],
                "cfp_keywords": case["expected_cfp_keywords"].split(";"),
                "cfp_message": (
                    f"I request proposals for {case['task_type']} with context "
                    f"{case['contextual_warning']}."
                ),
            }
        )

    system_prompt = (
        "You are an LLM-assisted initiator edge-cluster agent in LLM-MR-CNP. "
        "Return only valid JSON with keys cfp_intent, cfp_keywords, cfp_message."
    )
    return llm.complete(system_prompt, prompt)


def _complete_responder(
    llm: CompletionLLM | None,
    prompt: str,
    candidate: dict[str, str],
) -> str:
    if llm is None:
        return json.dumps(
            {
                "decision": candidate["expected_action"],
                "reason": candidate["expected_reason"],
                "natural_language": (
                    "I accept this CFP."
                    if candidate["expected_action"] == "accept"
                    else "I refuse this CFP because it is unsafe."
                ),
            }
        )

    system_prompt = (
        "You are an LLM-assisted responder edge-cluster agent in LLM-MR-CNP. "
        "Return only valid JSON with keys decision, reason, natural_language."
    )
    return llm.complete(system_prompt, prompt)


def _build_responder_prompt(
    strategy: str,
    case: dict[str, str],
    candidate: dict[str, str],
    cfp: dict,
    round_id: int,
) -> str:
    cfp_keywords = cfp.get("cfp_keywords", [])
    if isinstance(cfp_keywords, list):
        cfp_keywords_text = ", ".join(str(item) for item in cfp_keywords)
    else:
        cfp_keywords_text = str(cfp_keywords)

    refinement = ""
    if round_id > 1:
        refinement = (
            "\nRefinement request from initiator:\n"
            "Please refine your proposal using the extended workload context. "
            f"Task profile: {case['task_profile']} "
            f"Contextual warning: {case['contextual_warning']}"
        )

    strategy_hint = _strategy_hint(strategy)
    return f"""You are responder agent {candidate['candidate_agent']} in a multi-round Contract-Net negotiation.
Your represented cluster profile is: {candidate['candidate_cluster_type']}.

Received CFP from initiator:
Intent: {cfp.get('cfp_intent', 'unknown')}
Keywords: {cfp_keywords_text}
Message: {cfp.get('cfp_message', '')}
{refinement}

{strategy_hint}

Allowed decisions: accept, refuse.
Allowed reasons: resource_feasible, best_utility_feasible, latency_risk, resource_overload, bandwidth_risk, data_drift, conflict_risk, priority_requirement, contextual_risk.

Return only valid JSON:
{{
  "decision": "accept or refuse",
  "reason": "one allowed reason",
  "natural_language": "short negotiation response"
}}"""


def _strategy_hint(strategy: str) -> str:
    if strategy == "zero_shot":
        return "Decide directly from the CFP and your cluster profile."
    if strategy == "few_shot":
        return (
            "Example: if the CFP is latency-critical and your profile cannot preserve "
            "low latency, refuse with latency_risk. If your profile matches the CFP "
            "and no warning indicates risk, accept with resource_feasible."
        )
    if strategy == "cot":
        return "Think step by step internally about feasibility, QoS risk, and warnings, then return only JSON."
    if strategy == "react":
        return (
            "Use ReAct internally: Thought, Action=check feasibility/warning, "
            "Observation, then final JSON."
        )
    if strategy == "cot_few_shot":
        return (
            "Use the few-shot rule examples and think step by step internally before "
            "returning only JSON."
        )
    if strategy == "react_few_shot":
        return (
            "Use examples plus ReAct internally. Example: warning about data drift -> "
            "Action validate_contextual_warning -> refuse with data_drift."
        )
    if strategy == "react_few_shot_cot":
        return (
            "Use examples, step-by-step internal reasoning, and ReAct checks before "
            "returning only JSON."
        )
    return "Decide from the CFP and your cluster profile."


def _parse_initiator_response(raw_response: str) -> dict:
    data = _parse_json(raw_response)
    return {
        "cfp_intent": str(data.get("cfp_intent", "invalid")),
        "cfp_keywords": data.get("cfp_keywords", []),
        "cfp_message": str(data.get("cfp_message", raw_response.strip()[:300])),
    }


def _parse_responder_response(raw_response: str) -> dict[str, str]:
    data = _parse_json(raw_response)
    return {
        "decision": _normalize_decision(str(data.get("decision", "invalid"))),
        "reason": str(data.get("reason", "invalid")).lower().strip(),
        "natural_language": str(data.get("natural_language", "")),
    }


def _parse_json(raw_response: str) -> dict:
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _normalize_decision(value: str) -> str:
    value = value.lower().strip()
    if "accept" in value:
        return "accept"
    if "reject" in value or "refuse" in value or "decline" in value:
        return "refuse"
    return value if value in {"accept", "refuse"} else "invalid"


def _keyword_coverage(predicted_keywords, expected_keywords: str) -> float:
    expected = {item.lower() for item in expected_keywords.split(";") if item}
    if not expected:
        return 1.0
    if isinstance(predicted_keywords, str):
        predicted = {item.lower() for item in re.findall(r"[A-Za-z0-9_]+", predicted_keywords)}
    elif isinstance(predicted_keywords, list):
        predicted = {str(item).lower() for item in predicted_keywords}
    else:
        predicted = set()
    return len(expected & predicted) / len(expected)


def _consensus_reached(
    best_utility: float,
    second_utility: float | None,
    previous_best_utility: float | None,
    consensus_margin: float,
    epsilon: float,
    round_id: int,
) -> bool:
    if second_utility is None:
        return True
    if best_utility - second_utility >= consensus_margin:
        return True
    if previous_best_utility is not None and abs(best_utility - previous_best_utility) < epsilon:
        return True
    return False


def _responder_action_accuracy(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["responder_action_correct"] == "True") / len(rows)


def _metrics(rows: list[dict[str, str]]) -> InitiatorPromptingMetrics:
    cases = len(rows)
    intent_accuracy = _ratio(rows, "cfp_intent_correct")
    keyword_coverage = mean(float(row["cfp_keyword_coverage"]) for row in rows) if rows else 0.0
    cfp_accuracy = mean(float(row["cfp_accuracy"]) for row in rows) if rows else 0.0
    offloading_accuracy = _ratio(rows, "offloading_correct")
    responder_action_accuracy = (
        mean(float(row["responder_action_accuracy"]) for row in rows) if rows else 0.0
    )
    decision_times = [float(row["decision_time_ms"]) for row in rows]
    rounds = [float(row["negotiation_rounds"]) for row in rows]
    prompt_tokens = sum(int(row["prompt_tokens_est"]) for row in rows)
    completion_tokens = sum(int(row["completion_tokens_est"]) for row in rows)
    return InitiatorPromptingMetrics(
        cases=cases,
        cfp_intent_accuracy=intent_accuracy,
        cfp_keyword_coverage=keyword_coverage,
        cfp_accuracy=cfp_accuracy,
        offloading_accuracy=offloading_accuracy,
        responder_action_accuracy=responder_action_accuracy,
        avg_negotiation_rounds=mean(rounds) if rounds else 0.0,
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
    path = OUTPUT_DIR / "scenario4b_negotiation_details.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(results: list[InitiatorPromptingResult]) -> None:
    path = OUTPUT_DIR / "scenario4b_negotiation_summary.csv"
    fields = [
        "strategy",
        "llm_label",
        "cases",
        "cfp_intent_accuracy",
        "cfp_keyword_coverage",
        "cfp_accuracy",
        "offloading_accuracy",
        "responder_action_accuracy",
        "avg_negotiation_rounds",
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
                    "cfp_intent_accuracy": f"{m.cfp_intent_accuracy:.4f}",
                    "cfp_keyword_coverage": f"{m.cfp_keyword_coverage:.4f}",
                    "cfp_accuracy": f"{m.cfp_accuracy:.4f}",
                    "offloading_accuracy": f"{m.offloading_accuracy:.4f}",
                    "responder_action_accuracy": f"{m.responder_action_accuracy:.4f}",
                    "avg_negotiation_rounds": f"{m.avg_negotiation_rounds:.2f}",
                    "avg_decision_time_ms": f"{m.avg_decision_time_ms:.2f}",
                    "prompt_tokens_est": m.prompt_tokens_est,
                    "completion_tokens_est": m.completion_tokens_est,
                    "total_tokens_est": m.total_tokens_est,
                }
            )


def _estimate_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
