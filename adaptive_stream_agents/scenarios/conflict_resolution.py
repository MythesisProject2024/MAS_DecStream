from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from adaptive_stream_agents.core.agent import EdgeClusterAgent
from adaptive_stream_agents.core.domain import ClusterState, StreamTask
from adaptive_stream_agents.llm.adapters import DeterministicJsonLLM
from adaptive_stream_agents.graph.benchmark_workflow import run_benchmark_graph


CRITICALITY_WEIGHT = {"low": 1.0, "medium": 1.5, "high": 2.0, "emergency": 3.0}


@dataclass(frozen=True)
class OffloadingRequest:
    source_agent_id: str
    task: StreamTask


@dataclass(frozen=True)
class Allocation:
    task_id: str
    source_agent_id: str
    selected_agent_id: str | None
    accepted: bool
    confidence: float
    reason: str


@dataclass(frozen=True)
class ConflictMetrics:
    conflict_resolution_rate: float
    resource_overcommitment_rate: float
    qos_violation_rate: float
    latency_deadline_violation_rate: float
    critical_task_acceptance_rate: float
    load_balance_index: float
    global_utility: float
    negotiation_rounds: int
    communication_messages: int
    collaboration_cost: float
    collaboration_time_ms: float
    cluster_loads: Mapping[str, float]


@dataclass(frozen=True)
class ConflictScenario:
    agents: list[EdgeClusterAgent]
    requests: list[OffloadingRequest]


@dataclass(frozen=True)
class ConflictBenchmarkResult:
    method_name: str
    allocations: list[Allocation]
    metrics: ConflictMetrics
    llm_label: str | None = None


def build_conflict_resolution_scenario(
    num_agents: int = 5,
    num_requests: int = 3,
) -> ConflictScenario:
    if num_agents < 5:
        raise ValueError("Scenario 3 requires at least 5 agents.")
    if num_requests < 3:
        raise ValueError("Scenario 3 requires at least 3 concurrent requests.")

    states = [
        ClusterState("cluster-0", 64, 60, 128, 106, 50, 44, 38, 0.25, 0.10, 0.70, 0.30),
        ClusterState("cluster-1", 110, 42, 192, 80, 90, 36, 27, 0.12, 0.08, 0.90, 0.22),
        ClusterState("cluster-2", 96, 88, 160, 136, 75, 62, 42, 0.35, 0.20, 0.68, 0.33),
        ClusterState("cluster-3", 96, 58, 192, 102, 85, 49, 31, 0.18, 0.10, 0.84, 0.27),
        ClusterState("cluster-4", 128, 118, 256, 230, 95, 84, 35, 0.20, 0.12, 0.80, 0.40),
    ]
    states.extend(_generate_additional_states(num_agents - len(states)))
    agents = [EdgeClusterAgent(f"A{index}", state) for index, state in enumerate(states)]

    requests = [
        OffloadingRequest(
            "A0",
            StreamTask("ECG-CR", "ECG", 30.0, 18.0, 12.0, 45, "high", 0.85),
        ),
        OffloadingRequest(
            "A2",
            StreamTask("VID-CR", "Video surveillance", 45.0, 42.0, 30.0, 85, "medium", 0.70),
        ),
        OffloadingRequest(
            "A4",
            StreamTask("ALARM-CR", "Emergency alarm", 26.0, 14.0, 8.0, 35, "emergency", 0.90),
        ),
    ]
    requests.extend(_generate_additional_requests(num_requests - len(requests), num_agents))
    return ConflictScenario(agents=agents, requests=requests[:num_requests])


def _generate_additional_states(count: int) -> list[ClusterState]:
    states: list[ClusterState] = []
    for offset in range(count):
        index = offset + 5
        profile = index % 5
        if profile == 0:
            state = ClusterState(
                f"cluster-{index}", 112, 55, 192, 82, 88, 35, 26, 0.10, 0.08, 0.91, 0.20
            )
        elif profile == 1:
            state = ClusterState(
                f"cluster-{index}", 96, 66, 160, 96, 80, 46, 32, 0.18, 0.12, 0.84, 0.25
            )
        elif profile == 2:
            state = ClusterState(
                f"cluster-{index}", 128, 82, 224, 132, 95, 54, 30, 0.15, 0.10, 0.88, 0.27
            )
        elif profile == 3:
            state = ClusterState(
                f"cluster-{index}", 88, 74, 144, 116, 70, 54, 40, 0.30, 0.18, 0.72, 0.34
            )
        else:
            state = ClusterState(
                f"cluster-{index}", 140, 92, 256, 164, 105, 62, 34, 0.22, 0.14, 0.80, 0.31
            )
        states.append(state)
    return states


def _generate_additional_requests(count: int, num_agents: int) -> list[OffloadingRequest]:
    templates = [
        ("AI-INF", "AI inference", 38.0, 30.0, 20.0, 55, "high", 0.82),
        ("THERM", "Thermal monitoring", 20.0, 16.0, 10.0, 65, "medium", 0.55),
        ("AMB", "Ambulance tracking", 32.0, 18.0, 14.0, 40, "emergency", 0.88),
        ("CAM", "Camera analytics", 42.0, 34.0, 28.0, 80, "medium", 0.68),
        ("FALL", "Fall detection", 28.0, 20.0, 12.0, 45, "high", 0.76),
        ("FIRE", "Fire alarm", 30.0, 18.0, 10.0, 35, "emergency", 0.92),
        ("PARK", "Parking stream", 34.0, 24.0, 18.0, 70, "medium", 0.62),
        ("ECG2", "ECG", 30.0, 18.0, 12.0, 45, "high", 0.84),
        ("VID2", "Video surveillance", 45.0, 42.0, 30.0, 85, "medium", 0.70),
    ]
    requests: list[OffloadingRequest] = []
    source_candidates = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    for index in range(count):
        task_id, stream_type, cpu, memory, bandwidth, latency, criticality, dynamicity = (
            templates[index % len(templates)]
        )
        source_index = source_candidates[index % len(source_candidates)] % num_agents
        requests.append(
            OffloadingRequest(
                f"A{source_index}",
                StreamTask(
                    f"{task_id}-CR-{index + 4}",
                    stream_type,
                    cpu,
                    memory,
                    bandwidth,
                    latency,
                    criticality,
                    dynamicity,
                ),
            )
        )
    return requests


def run_conflict_benchmark(
    scenario: ConflictScenario, m3_llm_label: str = "mock"
) -> list[ConflictBenchmarkResult]:
    return [
        _run_classic_contract_net(scenario),
        _run_rule_based_multi_round(scenario),
        _run_llm_assisted_multi_round(scenario, m3_llm_label),
    ]


def format_conflict_results(results: list[ConflictBenchmarkResult]) -> str:
    lines = ["Scenario 3: Conflict Resolution Benchmark", "=" * 43]
    for result in results:
        lines.extend(["", result.method_name, "-" * len(result.method_name)])
        if result.llm_label:
            lines.append(f"LLM used by M3 agents: {result.llm_label}")
        for allocation in result.allocations:
            selected = allocation.selected_agent_id or "NONE"
            lines.append(
                f"{allocation.task_id} from {allocation.source_agent_id} -> {selected} "
                f"(accepted={allocation.accepted}, confidence={allocation.confidence:.2f})"
            )
            lines.append(f"  reason: {allocation.reason}")
        metrics = result.metrics
        lines.extend(
            [
                "Metrics:",
                f"  conflict_resolution_rate={metrics.conflict_resolution_rate:.2f}",
                f"  resource_overcommitment_rate={metrics.resource_overcommitment_rate:.2f}",
                f"  qos_violation_rate={metrics.qos_violation_rate:.2f}",
                f"  latency_deadline_violation_rate={metrics.latency_deadline_violation_rate:.2f}",
                f"  critical_task_acceptance_rate={metrics.critical_task_acceptance_rate:.2f}",
                f"  load_balance_index={metrics.load_balance_index:.2f}",
                f"  global_utility={metrics.global_utility:.2f}",
                f"  negotiation_rounds={metrics.negotiation_rounds}",
                f"  communication_messages={metrics.communication_messages}",
                f"  collaboration_cost={metrics.collaboration_cost:.2f}",
                f"  collaboration_time_ms={metrics.collaboration_time_ms:.2f}",
                "  cluster_loads="
                + ", ".join(
                    f"{agent_id}:{load:.2f}" for agent_id, load in metrics.cluster_loads.items()
                ),
            ]
        )
    return "\n".join(lines)


def _run_classic_contract_net(scenario: ConflictScenario) -> ConflictBenchmarkResult:
    allocations = [
        _select_best_without_reservation(scenario.agents, request)
        for request in scenario.requests
    ]
    return ConflictBenchmarkResult(
        "M1: LangGraph + Rule-Based Classic Contract-Net",
        allocations,
        _compute_metrics(scenario, allocations, negotiation_rounds=1),
    )


def _run_rule_based_multi_round(scenario: ConflictScenario) -> ConflictBenchmarkResult:
    allocations = _allocate_with_reservation(
        scenario,
        ordered_requests=scenario.requests,
        qualitative_priority=False,
    )
    return ConflictBenchmarkResult(
        "M2: LangGraph + Rule-Based Multi-Round Contract-Net",
        allocations,
        _compute_metrics(scenario, allocations, negotiation_rounds=2),
    )


def _run_llm_assisted_multi_round(
    scenario: ConflictScenario, llm_label: str
) -> ConflictBenchmarkResult:
    ordered = sorted(
        scenario.requests,
        key=lambda request: CRITICALITY_WEIGHT[request.task.criticality],
        reverse=True,
    )
    allocations = _allocate_with_reservation(
        scenario,
        ordered_requests=ordered,
        qualitative_priority=True,
    )
    return ConflictBenchmarkResult(
        "M3: LangGraph + LLM-Assisted Multi-Round Contract-Net",
        allocations,
        _compute_metrics(scenario, allocations, negotiation_rounds=2),
        llm_label,
    )


def _select_best_without_reservation(
    agents: list[EdgeClusterAgent], request: OffloadingRequest
) -> Allocation:
    selected = run_benchmark_graph(
        {"agents": agents, "request": request},
        lambda context: [
            EdgeClusterAgent(agent.agent_id, agent.state).evaluate_cfp(
                context["request"].task, round_id=1
            )
            for agent in context["agents"]
            if agent.agent_id != context["request"].source_agent_id
        ],
        lambda _context, proposals: [proposal for proposal in proposals if proposal.accepted],
        lambda _context, proposals: (
            max(proposals, key=lambda proposal: proposal.confidence)
            if proposals
            else None
        ),
    )
    if selected is None:
        return Allocation(
            request.task.task_id,
            request.source_agent_id,
            None,
            False,
            0.0,
            "No candidate accepted the task.",
        )
    return Allocation(
        request.task.task_id,
        request.source_agent_id,
        selected.agent_id,
        True,
        selected.confidence,
        "Single-round award; no reservation check against simultaneous awards.",
    )


def _allocate_with_reservation(
    scenario: ConflictScenario,
    ordered_requests: list[OffloadingRequest],
    qualitative_priority: bool,
) -> list[Allocation]:
    remaining = {agent.agent_id: agent.state for agent in scenario.agents}
    allocations: list[Allocation] = []

    for request in ordered_requests:
        candidates = [
            agent for agent in scenario.agents if agent.agent_id != request.source_agent_id
        ]
        result = run_benchmark_graph(
            {
                "candidates": candidates,
                "remaining": remaining,
                "request": request,
                "use_llm": qualitative_priority,
            },
            _conflict_proposal_node,
            _conflict_validation_node,
            _conflict_selection_node,
        )

        if result is None:
            allocations.append(
                Allocation(
                    request.task.task_id,
                    request.source_agent_id,
                    None,
                    False,
                    0.0,
                    "No remaining feasible host after conflict-aware reservation.",
                )
            )
            continue

        selected, proposal, objective_utility = result
        remaining[selected.agent_id] = _reserve(remaining[selected.agent_id], request.task)
        reason = "Refinement reallocates tasks using reserved resources."
        if qualitative_priority:
            reason = (
                "LLM proposal reasoning: "
                + proposal.natural_language
                + f" Hard validation retained this host and objective utility "
                f"({objective_utility:.2f}) selected it; LLM confidence was not used for ranking."
            )
        allocations.append(
            Allocation(
                request.task.task_id,
                request.source_agent_id,
                selected.agent_id,
                True,
                proposal.confidence,
                reason,
            )
        )

    return sorted(allocations, key=lambda allocation: allocation.task_id)


def _conflict_proposal_node(context: dict) -> list:
    proposals = []
    request = context["request"]
    for agent in context["candidates"]:
        state = context["remaining"][agent.agent_id]
        llm = agent.llm if context["use_llm"] else None
        if context["use_llm"] and llm is None:
            llm = DeterministicJsonLLM()
        proposal = EdgeClusterAgent(agent.agent_id, state, llm=llm).evaluate_cfp(
            request.task, round_id=2
        )
        proposals.append((agent, state, proposal))
    return proposals


def _conflict_validation_node(context: dict, proposals: list) -> list:
    task = context["request"].task
    return [
        (agent, state, proposal)
        for agent, state, proposal in proposals
        if proposal.accepted and _state_can_host(state, task)
    ]


def _conflict_selection_node(context: dict, proposals: list):
    if not proposals:
        return None
    task = context["request"].task
    ranked = [
        (_allocation_objective(state, task), agent, proposal)
        for agent, state, proposal in proposals
    ]
    utility, selected, proposal = max(ranked, key=lambda item: item[0])
    return selected, proposal, utility


def _allocation_objective(state: ClusterState, task: StreamTask) -> float:
    cpu_margin = (state.cpu_available - task.cpu_demand) / state.cpu_capacity
    memory_margin = (state.memory_available - task.memory_demand) / state.memory_capacity
    bandwidth_margin = (
        state.bandwidth_available - task.bandwidth_demand
    ) / state.bandwidth_capacity
    latency_margin = (task.latency_budget_ms - state.avg_latency_ms) / task.latency_budget_ms
    return (
        0.30 * cpu_margin
        + 0.20 * memory_margin
        + 0.20 * bandwidth_margin
        + 0.20 * latency_margin
        + 0.10 * state.stability_score
    )


def _compute_metrics(
    scenario: ConflictScenario,
    allocations: list[Allocation],
    negotiation_rounds: int,
) -> ConflictMetrics:
    assigned = [allocation for allocation in allocations if allocation.selected_agent_id]
    overcommitted_hosts = _overcommitted_hosts(scenario, assigned)
    violated = [
        allocation
        for allocation in assigned
        if allocation.selected_agent_id in overcommitted_hosts
        or _latency_violation(scenario, allocation)
    ]
    critical_requests = [
        request
        for request in scenario.requests
        if request.task.criticality in {"high", "emergency"}
    ]
    accepted_critical = [
        allocation
        for allocation in assigned
        if _task_by_id(scenario, allocation.task_id).criticality in {"high", "emergency"}
        and allocation.selected_agent_id not in overcommitted_hosts
    ]

    conflict_resolution_rate = 1.0 if not overcommitted_hosts else 0.0
    resource_overcommitment_rate = len(overcommitted_hosts) / max(len(scenario.agents), 1)
    qos_violation_rate = len(violated) / max(len(assigned), 1)
    latency_deadline_violation_rate = sum(
        1 for allocation in assigned if _latency_violation(scenario, allocation)
    ) / max(len(assigned), 1)
    critical_task_acceptance_rate = len(accepted_critical) / max(len(critical_requests), 1)
    cluster_loads = _cluster_loads(scenario, assigned)
    load_balance_index = _jain_index(list(cluster_loads.values()))
    global_utility = sum(
        CRITICALITY_WEIGHT[_task_by_id(scenario, allocation.task_id).criticality]
        * allocation.confidence
        for allocation in assigned
        if allocation.selected_agent_id not in overcommitted_hosts
    ) - 2.0 * len(violated)
    communication_messages = (
        len(scenario.requests) * max(len(scenario.agents) - 1, 1) * negotiation_rounds
    )
    collaboration_cost = float(communication_messages)
    collaboration_time_ms = 100.0 * negotiation_rounds + 5.0 * communication_messages

    return ConflictMetrics(
        conflict_resolution_rate,
        resource_overcommitment_rate,
        qos_violation_rate,
        latency_deadline_violation_rate,
        critical_task_acceptance_rate,
        load_balance_index,
        global_utility,
        negotiation_rounds,
        communication_messages,
        collaboration_cost,
        collaboration_time_ms,
        MappingProxyType(cluster_loads),
    )


def _state_can_host(state: ClusterState, task: StreamTask) -> bool:
    return (
        state.cpu_available >= task.cpu_demand
        and state.memory_available >= task.memory_demand
        and state.bandwidth_available >= task.bandwidth_demand
        and state.avg_latency_ms <= task.latency_budget_ms
    )


def _reserve(state: ClusterState, task: StreamTask) -> ClusterState:
    return ClusterState(
        state.cluster_id,
        state.cpu_capacity,
        state.cpu_used + task.cpu_demand,
        state.memory_capacity,
        state.memory_used + task.memory_demand,
        state.bandwidth_capacity,
        state.bandwidth_used + task.bandwidth_demand,
        state.avg_latency_ms,
        state.predicted_cpu_growth,
        state.predicted_memory_growth,
        state.stability_score,
        state.energy_pressure,
    )


def _overcommitted_hosts(
    scenario: ConflictScenario, allocations: list[Allocation]
) -> set[str]:
    used = {
        agent.agent_id: [agent.state.cpu_used, agent.state.memory_used, agent.state.bandwidth_used]
        for agent in scenario.agents
    }
    capacity = {
        agent.agent_id: [
            agent.state.cpu_capacity,
            agent.state.memory_capacity,
            agent.state.bandwidth_capacity,
        ]
        for agent in scenario.agents
    }
    for allocation in allocations:
        task = _task_by_id(scenario, allocation.task_id)
        used[allocation.selected_agent_id][0] += task.cpu_demand
        used[allocation.selected_agent_id][1] += task.memory_demand
        used[allocation.selected_agent_id][2] += task.bandwidth_demand

    return {
        agent_id
        for agent_id, values in used.items()
        if any(value > limit for value, limit in zip(values, capacity[agent_id]))
    }


def _cluster_loads(scenario: ConflictScenario, allocations: list[Allocation]) -> dict[str, float]:
    used = {
        agent.agent_id: [agent.state.cpu_used, agent.state.memory_used, agent.state.bandwidth_used]
        for agent in scenario.agents
    }
    capacity = {
        agent.agent_id: [
            agent.state.cpu_capacity,
            agent.state.memory_capacity,
            agent.state.bandwidth_capacity,
        ]
        for agent in scenario.agents
    }
    for allocation in allocations:
        task = _task_by_id(scenario, allocation.task_id)
        used[allocation.selected_agent_id][0] += task.cpu_demand
        used[allocation.selected_agent_id][1] += task.memory_demand
        used[allocation.selected_agent_id][2] += task.bandwidth_demand

    return {
        agent_id: sum(value / limit for value, limit in zip(used[agent_id], capacity[agent_id])) / 3.0
        for agent_id in used
    }


def _jain_index(values: list[float]) -> float:
    squared_sum = sum(value * value for value in values)
    if squared_sum == 0:
        return 1.0
    return (sum(values) ** 2) / (len(values) * squared_sum)


def _latency_violation(scenario: ConflictScenario, allocation: Allocation) -> bool:
    task = _task_by_id(scenario, allocation.task_id)
    agent = next(agent for agent in scenario.agents if agent.agent_id == allocation.selected_agent_id)
    return agent.state.avg_latency_ms > task.latency_budget_ms


def _task_by_id(scenario: ConflictScenario, task_id: str) -> StreamTask:
    return next(request.task for request in scenario.requests if request.task.task_id == task_id)
