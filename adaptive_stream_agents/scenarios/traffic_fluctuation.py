from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from adaptive_stream_agents.core.agent import EdgeClusterAgent
from adaptive_stream_agents.core.domain import ClusterState, StreamTask
from adaptive_stream_agents.graph.langgraph_contractnet import ContractNetConfig, LangGraphContractNetSession


WINDOW_SECONDS = 10
EXECUTION_SECONDS = 300
WINDOW_COUNT = EXECUTION_SECONDS // WINDOW_SECONDS
APP_DEADLINE_MS = 115.0
REMOTE_TRANSFER_MS = 8.0
OVERLOAD_THRESHOLD = 0.86
CFP_OUTPUT_PATH = Path("outputs") / "scenario1_traffic_fluctuation_cfps.csv"
NEGOTIATION_OUTPUT_PATH = Path("outputs") / "scenario1_traffic_fluctuation_negotiation_rounds.csv"


@dataclass(frozen=True)
class TrafficWorkload:
    record_id: str
    stream_task_type: str
    gpu_request: float
    bandwidth_request: float
    semantic_requirements: str
    qualitative_warnings: str


@dataclass(frozen=True)
class TrafficClusterWindow:
    record_id: str
    agent_id: str
    cluster_id: str
    gpu_used: float
    cpu_used: float
    memory_used: float
    bandwidth_used: float
    latency_ms: float
    energy_pressure: float
    trust_score: float
    low_latency_tier: bool
    predicted_gpu: tuple[float, ...]
    predicted_cpu: tuple[float, ...]
    qualitative_warnings: str


@dataclass(frozen=True)
class TrafficWindow:
    index: int
    workload: TrafficWorkload
    clusters: dict[str, TrafficClusterWindow]


@dataclass(frozen=True)
class TrafficFluctuationScenario:
    agents: list[EdgeClusterAgent]
    task: StreamTask
    initiator_id: str
    windows: list[TrafficWindow]


@dataclass(frozen=True)
class TrafficFluctuationMetrics:
    adaptation_triggers: int
    successful_migration_rate: float
    application_latency_violation_rate: float
    average_application_latency_ms: float
    global_utility: float
    load_balancing_degree: float
    negotiation_rounds: float
    collaboration_cost: float
    collaboration_time_ms: float
    prompt_tokens_est: int
    completion_tokens_est: int
    total_tokens_est: int


@dataclass(frozen=True)
class TrafficFluctuationResult:
    method_name: str
    metrics: TrafficFluctuationMetrics
    reason: str
    llm_label: str | None = None
    cfp_examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class NegotiationPromptTrace:
    initial_cfp: str
    refinement: str
    prompt_tokens: int
    completion_tokens: int


def build_traffic_fluctuation_scenario() -> TrafficFluctuationScenario:
    base_path = Path(__file__).resolve().parents[2] / "datasets"
    workloads = _read_workloads(base_path / "alibaba_job_summary_augmented_v2_workloads.csv")
    snapshots = _read_cluster_snapshots(
        base_path / "alibaba_job_summary_augmented_v2_cluster_snapshots.csv"
    )

    windows: list[TrafficWindow] = []
    for index, record_id in enumerate(list(workloads)[:WINDOW_COUNT]):
        clusters = snapshots[record_id]
        windows.append(TrafficWindow(index=index, workload=workloads[record_id], clusters=clusters))

    first_clusters = windows[0].clusters
    agents = [
        EdgeClusterAgent(agent_id, _state_from_window(cluster))
        for agent_id, cluster in first_clusters.items()
    ]
    task = StreamTask(
        task_id="TRAFFIC-APP-CRITICAL",
        stream_type="Smart traffic prediction and recommendation DAG",
        cpu_demand=20.0,
        memory_demand=16.0,
        bandwidth_demand=16.0,
        latency_budget_ms=int(APP_DEADLINE_MS),
        criticality="high",
        dynamicity=0.85,
    )
    return TrafficFluctuationScenario(
        agents=agents,
        task=task,
        initiator_id="A0",
        windows=windows,
    )


def run_traffic_fluctuation_benchmark(
    scenario: TrafficFluctuationScenario,
    m3_llm_label: str = "mock",
) -> list[TrafficFluctuationResult]:
    results = [
        _run_method(scenario, "RB-SR-CNP", 1, m3_llm_label=None),
        _run_method(scenario, "RB-MR-CNP", 2, m3_llm_label=None),
        _run_method(scenario, "LLM-MR-CNP", 3, m3_llm_label=m3_llm_label),
    ]
    _write_cfp_output(scenario, m3_llm_label)
    return results


def format_traffic_fluctuation_results(results: list[TrafficFluctuationResult]) -> str:
    lines = [
        "Scenario 1: Dynamic Smart-Traffic Stream Offloading Under Workload Fluctuation",
        "=" * 78,
        f"Execution horizon: {EXECUTION_SECONDS}s, scheduling window: {WINDOW_SECONDS}s, windows: {WINDOW_COUNT}",
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


def _run_method(
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
    prompt_tokens = 0
    completion_tokens = 0
    successful_migrations = 0
    adaptation_triggers = 0
    collaboration_cost = 0.0

    for window in scenario.windows:
        current_host = scenario.initiator_id
        a0 = window.clusters[scenario.initiator_id]
        if not _adaptation_required(window, a0):
            selected_hosts.append(current_host)
            latencies.append(_application_latency(window, current_host))
            continue

        adaptation_triggers += 1
        collaboration_cost += 4.0 * rounds
        cfp = _formulate_cfp(window)
        if method == "LLM-MR-CNP":
            trace = _progressive_prompt_trace(window)
            prompt_tokens += trace.prompt_tokens
            completion_tokens += trace.completion_tokens
            if len(cfp_examples) < 3:
                cfp_examples.append(f"{trace.initial_cfp} | Refinement: {trace.refinement}")
        host = _select_host(window, method)
        if host is not None:
            successful_migrations += 1
            current_host = host
        selected_hosts.append(current_host)
        latencies.append(_application_latency(window, current_host))

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
        _method_reason(method),
        m3_llm_label,
        tuple(cfp_examples),
    )


def _run_langgraph_if_available(scenario: TrafficFluctuationScenario) -> None:
    try:
        LangGraphContractNetSession(
            agents=scenario.agents,
            task=scenario.task,
            initiator_id=scenario.initiator_id,
            config=ContractNetConfig(max_rounds=3, consensus_margin=0.15, epsilon=0.03, min_score=0.60),
        ).run()
    except RuntimeError as exc:
        if "LangGraph is not installed" not in str(exc):
            raise


def _write_cfp_output(scenario: TrafficFluctuationScenario, llm_label: str) -> None:
    CFP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    negotiation_rows: list[dict[str, str | int | float]] = []
    for window in scenario.windows:
        a0 = window.clusters[scenario.initiator_id]
        if not _adaptation_required(window, a0):
            continue
        trace = _progressive_prompt_trace(window)
        negotiation_rows.extend(_negotiation_rows_for_window(window, trace, llm_label))
        rows.append(
            {
                "scenario": "traffic-fluctuation",
                "method": "LLM-MR-CNP",
                "llm_label": llm_label,
                "window_index": window.index,
                "record_id": window.workload.record_id,
                "deadline_ms": APP_DEADLINE_MS,
                "a0_application_latency_ms": round(_application_latency(window, "A0"), 3),
                "prompt_tokens_est": trace.prompt_tokens,
                "completion_tokens_est": trace.completion_tokens,
                "total_tokens_est": trace.prompt_tokens + trace.completion_tokens,
                "formulated_cfp": trace.initial_cfp,
                "refinement_message": trace.refinement,
            }
        )

    with CFP_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "method",
                "llm_label",
                "window_index",
                "record_id",
                "deadline_ms",
                "a0_application_latency_ms",
                "prompt_tokens_est",
                "completion_tokens_est",
                "total_tokens_est",
                "formulated_cfp",
                "refinement_message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with NEGOTIATION_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "method",
                "llm_label",
                "window_index",
                "record_id",
                "round_id",
                "message_type",
                "sender",
                "receiver",
                "selected_host",
                "tokens_est",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(negotiation_rows)


def _negotiation_rows_for_window(
    window: TrafficWindow,
    trace: NegotiationPromptTrace,
    llm_label: str,
) -> list[dict[str, str | int | float]]:
    selected_host = _select_host(window, "LLM-MR-CNP")
    rows: list[dict[str, str | int | float]] = [
        _message_row(window, llm_label, 1, "cfp", "A0", "ALL", "", selected_host, trace.initial_cfp),
    ]

    accepted_round1: list[str] = []
    for agent_id in _candidate_agent_ids(window):
        message, accepted = _response_message(window, agent_id, prediction_aware=False)
        if accepted:
            accepted_round1.append(agent_id)
        rows.append(
            _message_row(
                window, llm_label, 1, "proposal", agent_id, "A0", accepted, selected_host, message
            )
        )

    refinement_receivers = ",".join(accepted_round1) if accepted_round1 else "NONE"
    rows.append(
        _message_row(
            window,
            llm_label,
            2,
            "refinement",
            "A0",
            refinement_receivers,
            "",
            selected_host,
            trace.refinement,
        )
    )
    for agent_id in accepted_round1:
        message, accepted = _response_message(window, agent_id, prediction_aware=True)
        rows.append(
            _message_row(
                window, llm_label, 2, "refined_proposal", agent_id, "A0", accepted, selected_host, message
            )
        )

    final_message = (
        f"Final decision from A0: migrate the selected smart-traffic stream task to "
        f"{selected_host or 'NONE'}. The decision is validated by hard constraints and "
        f"objective utility; LLM confidence is not used as the ranking criterion."
    )
    rows.append(
        _message_row(
            window, llm_label, 3, "final_decision", "A0", "ALL", "", selected_host, final_message
        )
    )
    return rows


def _message_row(
    window: TrafficWindow,
    llm_label: str,
    round_id: int,
    message_type: str,
    sender: str,
    receiver: str,
    accepted: bool | str,
    selected_host: str | None,
    message: str,
) -> dict[str, str | int | float]:
    return {
        "scenario": "traffic-fluctuation",
        "method": "LLM-MR-CNP",
        "llm_label": llm_label,
        "window_index": window.index,
        "record_id": window.workload.record_id,
        "round_id": round_id,
        "message_type": message_type,
        "sender": sender,
        "receiver": receiver,
        "selected_host": selected_host or "",
        "tokens_est": _estimate_tokens(message),
        "message": message,
    }


def _candidate_agent_ids(window: TrafficWindow) -> list[str]:
    return [agent_id for agent_id in window.clusters if agent_id != "A0"]


def _response_message(
    window: TrafficWindow,
    agent_id: str,
    prediction_aware: bool,
) -> tuple[str, bool]:
    task = window.workload
    cluster = window.clusters[agent_id]
    if prediction_aware:
        feasible = _prediction_feasible(window, agent_id) and _llm_context_safe(window, agent_id)
        risk = _predicted_qos_risk(window, agent_id)
        mode = "refined proposal"
    else:
        feasible = _current_feasible(window, agent_id)
        risk = _current_qos_risk(window, agent_id)
        mode = "initial proposal"

    decision = "accept" if feasible else "refuse"
    explanation = "I accept." if feasible else "I refuse."
    if prediction_aware and not _llm_context_safe(window, agent_id):
        explanation = "I refuse due to future contention risk."

    confidence = round(max(0.0, min(1.0, 1.0 - risk)), 3) if feasible else round(max(0.0, min(1.0, 0.45 - risk)), 3)
    natural_language = f"{explanation} QoS risk={risk:.2f}."
    return (
        json.dumps(
            {
                "confidence": confidence,
                "qos_risk": risk,
                "natural_language": natural_language,
            }
        ),
        feasible,
    )


def _current_qos_risk(window: TrafficWindow, agent_id: str) -> float:
    latency = _application_latency(window, agent_id)
    load = _load_after(window, window.clusters[agent_id])
    return round(max(0.0, 0.55 * (latency / APP_DEADLINE_MS) + 0.45 * load - 0.65), 3)


def _predicted_qos_risk(window: TrafficWindow, agent_id: str) -> float:
    cluster = window.clusters[agent_id]
    predicted_load = _predicted_load_after(window, cluster)
    predicted_latency = _latency_from_load(predicted_load, cluster, agent_id)
    return round(max(0.0, 0.60 * (predicted_latency / APP_DEADLINE_MS) + 0.40 * predicted_load - 0.65), 3)


def _select_host(window: TrafficWindow, method: str) -> str | None:
    candidates = [agent_id for agent_id in window.clusters if agent_id != "A0"]
    if method == "RB-SR-CNP":
        feasible = [agent_id for agent_id in candidates if _current_feasible(window, agent_id)]
        return max(feasible, key=lambda agent_id: _current_utility(window, agent_id), default=None)

    if method == "RB-MR-CNP":
        feasible = [agent_id for agent_id in candidates if _prediction_feasible(window, agent_id)]
        return max(feasible, key=lambda agent_id: _prediction_utility(window, agent_id), default=None)

    feasible = [
        agent_id
        for agent_id in candidates
        if _prediction_feasible(window, agent_id) and _llm_context_safe(window, agent_id)
    ]
    return max(feasible, key=lambda agent_id: _llm_validated_utility(window, agent_id), default=None)


def _progressive_prompt_trace(window: TrafficWindow) -> NegotiationPromptTrace:
    initial_cfp = _formulate_cfp(window)
    refinement = _formulate_refinement(window)
    prompt_tokens = _estimate_tokens(_llm_prompt_context(window, initial_cfp, refinement))
    completion_tokens = _estimate_tokens(initial_cfp) + _estimate_tokens(refinement)
    return NegotiationPromptTrace(initial_cfp, refinement, prompt_tokens, completion_tokens)


def _formulate_cfp(window: TrafficWindow) -> str:
    task = window.workload
    request_type = _request_type(window)
    return (
        f"CFP from A0 at negotiation window {window.index}. Request type: {request_type}. "
        f"I request proposals for offloading a critical smart-traffic stream-processing task. "
        f"Minimum requirements: GPU={task.gpu_request:.2f}, latency_deadline={APP_DEADLINE_MS:.0f} ms. "
        f"Please respond with acceptance/refusal and a short QoS-risk rationale."
    )


def _formulate_refinement(window: TrafficWindow) -> str:
    task = window.workload
    return (
        f"Refinement from A0 at negotiation window {window.index}: please refine your proposal "
        f"using the extended workload profile. task_type={task.stream_task_type}, "
        f"bandwidth={task.bandwidth_request:.2f}, semantic_constraints=[{task.semantic_requirements}], "
        f"contextual_warnings={task.qualitative_warnings or 'none'}. "
        f"Prioritize candidates that can preserve application latency and avoid future contention."
    )


def _request_type(window: TrafficWindow) -> str:
    if window.index == 0:
        return "initial offloading request"
    return "offloading or refinement request"


def _adaptation_trigger_reason(window: TrafficWindow, cluster: TrafficClusterWindow) -> str:
    current_latency = _application_latency(window, cluster.agent_id)
    predicted_peak = max(cluster.predicted_cpu[-3:] + cluster.predicted_gpu[-3:])
    reasons = []
    if current_latency > APP_DEADLINE_MS:
        reasons.append("the current application latency exceeds the deadline")
    if predicted_peak + _task_pressure(window) > OVERLOAD_THRESHOLD:
        reasons.append("the predicted workload pressure may overload the hosting cluster")
    if "resource contention" in window.workload.qualitative_warnings:
        reasons.append("the workload trace contains a resource-contention warning")
    return "; ".join(reasons) if reasons else "the local state is close to the adaptation threshold"


def _llm_prompt_context(window: TrafficWindow, cfp: str, refinement: str) -> str:
    a0 = window.clusters["A0"]
    return (
        "You are representative agent A0. Formulate a natural-language Contract-Net CFP. "
        "Use your private local state only to decide whether adaptation is needed; do not "
        "disclose CPU, GPU, memory, bandwidth, latency, or prediction values in the CFP. "
        "Use progressive disclosure: the initial CFP must be compact and include only the "
        "most important task features; refinement can reveal additional workload features "
        "only when needed to discriminate candidate clusters. "
        f"Local state: cpu={a0.cpu_used:.2f}, gpu={a0.gpu_used:.2f}, "
        f"memory={a0.memory_used:.2f}, bandwidth={a0.bandwidth_used:.2f}, "
        f"latency={a0.latency_ms:.1f} ms, predicted_cpu={a0.predicted_cpu}, "
        f"predicted_gpu={a0.predicted_gpu}. Draft CFP: {cfp}. Draft refinement: {refinement}"
    )


def _estimate_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _adaptation_required(window: TrafficWindow, cluster: TrafficClusterWindow) -> bool:
    current_latency = _application_latency(window, cluster.agent_id)
    predicted_peak = max(cluster.predicted_cpu[-3:] + cluster.predicted_gpu[-3:])
    return (
        current_latency > APP_DEADLINE_MS
        or predicted_peak + _task_pressure(window) > OVERLOAD_THRESHOLD
        or "resource contention" in window.workload.qualitative_warnings
    )


def _current_feasible(window: TrafficWindow, agent_id: str) -> bool:
    cluster = window.clusters[agent_id]
    return _load_after(window, cluster) <= 1.02


def _prediction_feasible(window: TrafficWindow, agent_id: str) -> bool:
    cluster = window.clusters[agent_id]
    predicted_load = _predicted_load_after(window, cluster)
    predicted_latency = _latency_from_load(predicted_load, cluster, agent_id)
    return predicted_load <= 0.90 and predicted_latency <= APP_DEADLINE_MS


def _llm_context_safe(window: TrafficWindow, agent_id: str) -> bool:
    cluster = window.clusters[agent_id]
    warnings = " ".join([window.workload.qualitative_warnings, cluster.qualitative_warnings]).lower()
    if "resource contention" in warnings and _predicted_load_after(window, cluster) > 0.88:
        return False
    if "abnormal" in warnings and not cluster.low_latency_tier and cluster.trust_score < 0.74:
        return False
    return True


def _current_utility(window: TrafficWindow, agent_id: str) -> float:
    cluster = window.clusters[agent_id]
    load = _load_after(window, cluster)
    immediate_capacity = 1.0 - load
    return 1.6 * immediate_capacity + 0.7 * (1.0 - cluster.gpu_used) - 0.15 * cluster.energy_pressure


def _prediction_utility(window: TrafficWindow, agent_id: str) -> float:
    cluster = window.clusters[agent_id]
    predicted_load = _predicted_load_after(window, cluster)
    predicted_latency = _latency_from_load(predicted_load, cluster, agent_id)
    return 2.3 * (1.0 - predicted_latency / APP_DEADLINE_MS) + 1.0 * (1.0 - predicted_load) - 0.2 * cluster.energy_pressure


def _llm_validated_utility(window: TrafficWindow, agent_id: str) -> float:
    cluster = window.clusters[agent_id]
    semantic_bonus = 0.22 if cluster.low_latency_tier else 0.0
    trust_bonus = 0.12 * cluster.trust_score
    return _prediction_utility(window, agent_id) + semantic_bonus + trust_bonus


def _application_latency(window: TrafficWindow, agent_id: str) -> float:
    cluster = window.clusters[agent_id]
    return _latency_from_load(_load_after(window, cluster), cluster, agent_id)


def _latency_from_load(load: float, cluster: TrafficClusterWindow, agent_id: str) -> float:
    remote_penalty = 0.0 if agent_id == "A0" else REMOTE_TRANSFER_MS
    low_latency_bonus = -7.0 if cluster.low_latency_tier else 0.0
    overload_penalty = 70.0 * max(0.0, load - OVERLOAD_THRESHOLD)
    return 42.0 + cluster.latency_ms * 0.55 + 50.0 * load + overload_penalty + remote_penalty + low_latency_bonus


def _load_after(window: TrafficWindow, cluster: TrafficClusterWindow) -> float:
    return min(1.5, _base_load(cluster) + _task_pressure(window))


def _predicted_load_after(window: TrafficWindow, cluster: TrafficClusterWindow) -> float:
    predicted_cpu = mean(cluster.predicted_cpu[-3:])
    predicted_gpu = mean(cluster.predicted_gpu[-3:])
    predicted_base = (
        0.30 * predicted_cpu
        + 0.25 * cluster.memory_used
        + 0.20 * cluster.bandwidth_used
        + 0.25 * predicted_gpu
    )
    return min(1.5, predicted_base + _task_pressure(window))


def _base_load(cluster: TrafficClusterWindow) -> float:
    return (
        0.30 * cluster.cpu_used
        + 0.25 * cluster.memory_used
        + 0.20 * cluster.bandwidth_used
        + 0.25 * cluster.gpu_used
    )


def _task_pressure(window: TrafficWindow) -> float:
    workload = window.workload
    semantic_pressure = 0.04 if "low_latency" in workload.semantic_requirements else 0.0
    return 0.12 + 0.16 * workload.gpu_request + 0.10 * workload.bandwidth_request + semantic_pressure


def _load_balancing_degree(windows: list[TrafficWindow], selected_hosts: list[str]) -> float:
    totals = {agent_id: 0.0 for agent_id in windows[0].clusters}
    for window, host in zip(windows, selected_hosts):
        for agent_id, cluster in window.clusters.items():
            totals[agent_id] += _base_load(cluster)
        totals[host] += _task_pressure(window)
    values = list(totals.values())
    avg = mean(values)
    if avg == 0:
        return 1.0
    imbalance = mean(abs(value - avg) for value in values) / avg
    return max(0.0, min(1.0, 1.0 - imbalance))


def _global_utility(
    violation_rate: float,
    success_rate: float,
    avg_latency: float,
    lbd: float,
    collaboration_cost: float,
) -> float:
    normalized_latency = avg_latency / APP_DEADLINE_MS
    return round(
        2.0 * success_rate
        + 1.5 * (1.0 - violation_rate)
        + 1.0 * lbd
        - 1.2 * normalized_latency
        - 0.003 * collaboration_cost,
        3,
    )


def _state_from_window(cluster: TrafficClusterWindow) -> ClusterState:
    return ClusterState(
        cluster_id=cluster.cluster_id,
        cpu_capacity=100.0,
        cpu_used=100.0 * cluster.cpu_used,
        memory_capacity=100.0,
        memory_used=100.0 * cluster.memory_used,
        bandwidth_capacity=100.0,
        bandwidth_used=100.0 * cluster.bandwidth_used,
        avg_latency_ms=cluster.latency_ms,
        predicted_cpu_growth=max(cluster.predicted_cpu[-1] - cluster.cpu_used, 0.0),
        predicted_memory_growth=0.0,
        stability_score=cluster.trust_score,
        energy_pressure=1.0 - cluster.energy_pressure,
    )


def _read_workloads(path: Path) -> dict[str, TrafficWorkload]:
    workloads: dict[str, TrafficWorkload] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            workloads[row["record_id"]] = TrafficWorkload(
                record_id=row["record_id"],
                stream_task_type=row["stream_task_type"],
                gpu_request=float(row["gpu_request_norm"]),
                bandwidth_request=float(row["bandwidth_request_norm"]),
                semantic_requirements=row["semantic_requirements"],
                qualitative_warnings=row["qualitative_warnings"],
            )
    return workloads


def _read_cluster_snapshots(path: Path) -> dict[str, dict[str, TrafficClusterWindow]]:
    snapshots: dict[str, dict[str, TrafficClusterWindow]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record_id = row["record_id"]
            cluster_number = int(row["cluster_id"][1:])
            agent_id = f"A{cluster_number - 1}"
            snapshots.setdefault(record_id, {})[agent_id] = TrafficClusterWindow(
                record_id=record_id,
                agent_id=agent_id,
                cluster_id=row["cluster_id"],
                gpu_used=float(row["current_gpu_used_norm"]),
                cpu_used=float(row["current_cpu_used_norm"]),
                memory_used=float(row["current_memory_used_norm"]),
                bandwidth_used=float(row["current_bandwidth_used_norm"]),
                latency_ms=float(row["current_latency_ms"]),
                energy_pressure=1.0 - float(row["energy_efficiency"]),
                trust_score=float(row["trust_score"]),
                low_latency_tier=_as_bool(row["low_latency_tier"]),
                predicted_gpu=_parse_series(row["predicted_gpu_used_norm"]),
                predicted_cpu=_parse_series(row["predicted_cpu_used_norm"]),
                qualitative_warnings=row["qualitative_warnings"],
            )
    return snapshots


def _parse_series(raw: str) -> tuple[float, ...]:
    return tuple(float(value) for value in raw.split(";") if value)


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() == "true"


def _method_label(method: str) -> str:
    labels = {
        "RB-SR-CNP": "RB-SR-CNP: LangGraph + Rule-Based Single-Round Contract-Net",
        "RB-MR-CNP": "RB-MR-CNP: LangGraph + Rule-Based Multi-Round Contract-Net",
        "LLM-MR-CNP": "LLM-MR-CNP: LangGraph + LLM-Assisted Multi-Round Contract-Net",
    }
    return labels[method]


def _method_reason(method: str) -> str:
    reasons = {
        "RB-SR-CNP": "The agent reacts to the current state and selects a feasible host in one round.",
        "RB-MR-CNP": "Agents refine proposals using predicted workload evolution before utility selection.",
        "LLM-MR-CNP": (
            "LLM agents interpret qualitative warnings and semantic context; hard constraints and "
            "objective utility select the final host."
        ),
    }
    return reasons[method]
