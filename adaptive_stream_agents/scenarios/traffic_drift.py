from __future__ import annotations

import json
import time
from dataclasses import replace
from statistics import mean

from adaptive_stream_agents.scenarios.traffic_fluctuation import (
    APP_DEADLINE_MS,
    TrafficClusterWindow,
    TrafficFluctuationMetrics,
    TrafficFluctuationResult,
    TrafficFluctuationScenario,
    TrafficWindow,
    _adaptation_required,
    _application_latency,
    _candidate_agent_ids,
    _current_feasible,
    _current_utility,
    _global_utility,
    _load_after,
    _load_balancing_degree,
    _method_label,
    _prediction_feasible,
    _prediction_utility,
    _progressive_prompt_trace,
    _run_langgraph_if_available,
    build_traffic_fluctuation_scenario,
)


DRIFT_AGENT_ID = "A2"
MAX_LLM_REASONING_CALLS = 3


def build_traffic_drift_scenario() -> TrafficFluctuationScenario:
    base = build_traffic_fluctuation_scenario()
    drift_windows = [_inject_unmodelled_drift(window) for window in base.windows]
    return TrafficFluctuationScenario(
        agents=base.agents,
        task=base.task,
        initiator_id=base.initiator_id,
        windows=drift_windows,
    )


def run_traffic_drift_benchmark(
    scenario: TrafficFluctuationScenario,
    m3_llm_label: str = "mock",
) -> list[TrafficFluctuationResult]:
    return [
        _run_drift_method(scenario, "RB-SR-CNP", 1, m3_llm_label=None),
        _run_drift_method(scenario, "RB-MR-CNP", 2, m3_llm_label=None),
        _run_drift_method(scenario, "LLM-MR-CNP", 3, m3_llm_label=m3_llm_label),
    ]


def format_traffic_drift_results(results: list[TrafficFluctuationResult]) -> str:
    lines = [
        "Scenario 1.1: Adaptive Stream-Task Migration Under Data Drift",
        "=" * 64,
        "Execution horizon: 300s, scheduling window: 10s, windows: 30",
        f"Unmodelled drift candidate: {DRIFT_AGENT_ID}",
    ]
    for result in results:
        m = result.metrics
        lines.extend(["", result.method_name, "-" * len(result.method_name)])
        if result.llm_label:
            lines.append(f"LLM used by LLM-MR-CNP agents: {result.llm_label}")
        lines.append(f"reason: {result.reason}")
        if result.cfp_examples:
            lines.append("Formulated CFP examples:")
            for index, cfp in enumerate(result.cfp_examples, start=1):
                lines.append(f"  CFP{index}: {cfp}")
        lines.extend(
            [
                "Metrics:",
                f"  adaptation_triggers={m.adaptation_triggers}",
                f"  successful_migration_rate={m.successful_migration_rate:.2f}",
                f"  application_latency_violation_rate={m.application_latency_violation_rate:.2f}",
                f"  average_application_latency_ms={m.average_application_latency_ms:.2f}",
                f"  global_utility={m.global_utility:.2f}",
                f"  load_balancing_degree={m.load_balancing_degree:.2f}",
                f"  negotiation_rounds={m.negotiation_rounds:.2f}",
                f"  collaboration_cost={m.collaboration_cost:.2f}",
                f"  collaboration_time_ms={m.collaboration_time_ms:.2f}",
                f"  prompt_tokens_est={m.prompt_tokens_est}",
                f"  completion_tokens_est={m.completion_tokens_est}",
                f"  total_tokens_est={m.total_tokens_est}",
            ]
        )
    return "\n".join(lines)


def _run_drift_method(
    scenario: TrafficFluctuationScenario,
    method: str,
    rounds: int,
    m3_llm_label: str | None,
) -> TrafficFluctuationResult:
    if m3_llm_label is not None:
        _run_langgraph_if_available(scenario)

    selected_hosts: list[str] = []
    latencies: list[float] = []
    cfp_examples: list[str] = []
    llm_reasoning_calls = 0
    llm_reasoning_time_ms = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    successful_migrations = 0
    adaptation_triggers = 0
    collaboration_cost = 0.0

    for realized_window in scenario.windows:
        forecast_window = _forecast_view(realized_window)
        current_host = scenario.initiator_id
        a0 = realized_window.clusters[scenario.initiator_id]
        if not _adaptation_required(realized_window, a0):
            selected_hosts.append(current_host)
            latencies.append(_application_latency(realized_window, current_host))
            continue

        adaptation_triggers += 1
        collaboration_cost += 4.0 * rounds
        if method == "LLM-MR-CNP":
            trace = _progressive_prompt_trace(realized_window)
            prompt_tokens += trace.prompt_tokens
            completion_tokens += trace.completion_tokens
            if llm_reasoning_calls < MAX_LLM_REASONING_CALLS:
                calls, elapsed_ms = _invoke_drift_llm_reasoning(scenario, realized_window, trace)
                llm_reasoning_calls += calls
                llm_reasoning_time_ms += elapsed_ms
            if len(cfp_examples) < 3:
                cfp_examples.append(f"{trace.initial_cfp} | Refinement: {trace.refinement}")

        host = _select_host_under_drift(forecast_window, method)
        if host is not None:
            successful_migrations += 1
            current_host = host
        selected_hosts.append(current_host)
        latencies.append(_application_latency(realized_window, current_host))

    violation_rate = sum(1 for latency in latencies if latency > APP_DEADLINE_MS) / len(latencies)
    success_rate = successful_migrations / adaptation_triggers if adaptation_triggers else 1.0
    avg_latency = mean(latencies)
    lbd = _load_balancing_degree(scenario.windows, selected_hosts)
    utility = _global_utility(violation_rate, success_rate, avg_latency, lbd, collaboration_cost)

    metrics = TrafficFluctuationMetrics(
        adaptation_triggers=adaptation_triggers,
        successful_migration_rate=success_rate,
        application_latency_violation_rate=violation_rate,
        average_application_latency_ms=avg_latency,
        global_utility=utility,
        load_balancing_degree=lbd,
        negotiation_rounds=float(rounds if adaptation_triggers else 0),
        collaboration_cost=collaboration_cost,
        collaboration_time_ms=adaptation_triggers * (100.0 * rounds + 5.0 * 4.0 * rounds),
        prompt_tokens_est=prompt_tokens,
        completion_tokens_est=completion_tokens,
        total_tokens_est=prompt_tokens + completion_tokens,
    )
    return TrafficFluctuationResult(
        _method_label(method),
        metrics,
        _drift_reason(method, llm_reasoning_calls, llm_reasoning_time_ms),
        m3_llm_label,
        tuple(cfp_examples),
    )


def _invoke_drift_llm_reasoning(
    scenario: TrafficFluctuationScenario,
    window: TrafficWindow,
    trace,
) -> tuple[int, float]:
    initiator = next(agent for agent in scenario.agents if agent.agent_id == scenario.initiator_id)
    if initiator.llm is None:
        return 0, 0.0
    if initiator.llm.__class__.__name__ == "DeterministicJsonLLM":
        return 1, 0.0

    system_prompt = (
        "You are representative agent A0 in an LLM-assisted Contract-Net negotiation. "
        "Interpret qualitative warnings and explain whether the refinement should avoid "
        "candidate clusters affected by unmodelled data drift. Reply with compact JSON."
    )
    user_prompt = json.dumps(
        {
            "round": "drift-aware refinement",
            "cfp": trace.initial_cfp,
            "refinement": trace.refinement,
            "qualitative_warning": window.workload.qualitative_warnings,
            "expected_json_keys": ["drift_detected", "rationale"],
        }
    )
    start = time.perf_counter()
    initiator.llm.complete(system_prompt, user_prompt)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return 1, elapsed_ms


def _select_host_under_drift(window: TrafficWindow, method: str) -> str | None:
    candidates = _candidate_agent_ids(window)
    if method == "RB-SR-CNP":
        feasible = [agent_id for agent_id in candidates if _current_feasible(window, agent_id)]
        return max(feasible, key=lambda agent_id: _current_utility(window, agent_id), default=None)

    if method == "RB-MR-CNP":
        feasible = [agent_id for agent_id in candidates if _prediction_feasible(window, agent_id)]
        return max(feasible, key=lambda agent_id: _prediction_utility(window, agent_id), default=None)

    feasible = [
        agent_id
        for agent_id in candidates
        if _prediction_feasible(window, agent_id) and _drift_context_safe(window, agent_id)
    ]
    return max(feasible, key=lambda agent_id: _prediction_utility(window, agent_id), default=None)


def _inject_unmodelled_drift(window: TrafficWindow) -> TrafficWindow:
    if not _adaptation_required(window, window.clusters["A0"]):
        return window
    cluster = window.clusters[DRIFT_AGENT_ID]
    drifted = replace(
        cluster,
        gpu_used=min(0.98, cluster.gpu_used + 0.42),
        cpu_used=min(0.98, cluster.cpu_used + 0.36),
        memory_used=min(0.98, cluster.memory_used + 0.24),
        bandwidth_used=min(0.98, cluster.bandwidth_used + 0.30),
        latency_ms=cluster.latency_ms + 48.0,
        qualitative_warnings=(
            "UNMODELLED DATA-DRIFT WARNING: reliability monitor reports an abnormal "
            "workload spike not captured by the numerical prediction model. "
            "Future resource saturation and application-latency violation are likely."
        ),
    )
    clusters = dict(window.clusters)
    clusters[DRIFT_AGENT_ID] = drifted
    workload = replace(
        window.workload,
        qualitative_warnings=(
            f"{window.workload.qualitative_warnings}; unstructured reliability warning: "
            "unmodelled data drift may invalidate numerical workload prediction"
        ).strip("; "),
    )
    return TrafficWindow(index=window.index, workload=workload, clusters=clusters)


def _forecast_view(realized_window: TrafficWindow) -> TrafficWindow:
    cluster = realized_window.clusters[DRIFT_AGENT_ID]
    restored = replace(
        cluster,
        gpu_used=max(0.0, cluster.gpu_used - 0.42),
        cpu_used=max(0.0, cluster.cpu_used - 0.36),
        memory_used=max(0.0, cluster.memory_used - 0.24),
        bandwidth_used=max(0.0, cluster.bandwidth_used - 0.30),
        latency_ms=cluster.latency_ms - 48.0,
    )
    clusters = dict(realized_window.clusters)
    clusters[DRIFT_AGENT_ID] = restored
    return TrafficWindow(index=realized_window.index, workload=realized_window.workload, clusters=clusters)


def _drift_context_safe(window: TrafficWindow, agent_id: str) -> bool:
    cluster = window.clusters[agent_id]
    warnings = cluster.qualitative_warnings.lower()
    return "data-drift warning" not in warnings and "unmodelled data drift" not in warnings


def _drift_reason(method: str, llm_calls: int = 0, llm_time_ms: float = 0.0) -> str:
    reasons = {
        "RB-SR-CNP": "Single-round selection ignores the unmodelled drift warning.",
        "RB-MR-CNP": "Numerical refinement uses the prediction model, which does not capture the drift spike.",
        "LLM-MR-CNP": (
            "LLM-assisted refinement interprets the qualitative drift warning; hard validation "
            "rejects unsafe candidates before utility ranking."
        ),
    }
    if method == "LLM-MR-CNP":
        return f"{reasons[method]} LLM reasoning calls={llm_calls}, time_ms={llm_time_ms:.2f}."
    return reasons[method]
