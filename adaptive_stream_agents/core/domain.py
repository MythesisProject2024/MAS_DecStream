from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamTask:
    """A stream-processing workload that may be migrated between edge clusters."""

    task_id: str
    stream_type: str
    cpu_demand: float
    memory_demand: float
    bandwidth_demand: float
    latency_budget_ms: int
    criticality: str
    dynamicity: float


@dataclass(frozen=True)
class ClusterState:
    """Local state visible to one representative edge node."""

    cluster_id: str
    cpu_capacity: float
    cpu_used: float
    memory_capacity: float
    memory_used: float
    bandwidth_capacity: float
    bandwidth_used: float
    avg_latency_ms: float
    predicted_cpu_growth: float
    predicted_memory_growth: float
    stability_score: float
    energy_pressure: float

    @property
    def cpu_available(self) -> float:
        return max(self.cpu_capacity - self.cpu_used, 0.0)

    @property
    def memory_available(self) -> float:
        return max(self.memory_capacity - self.memory_used, 0.0)

    @property
    def bandwidth_available(self) -> float:
        return max(self.bandwidth_capacity - self.bandwidth_used, 0.0)


@dataclass(frozen=True)
class Proposal:
    """Structured proposal produced from a natural-language local decision."""

    agent_id: str
    accepted: bool
    confidence: float
    utility: float
    rationale: str
    natural_language: str


@dataclass(frozen=True)
class NegotiationMessage:
    """Natural-language message exchanged between agents."""

    round_id: int
    sender: str
    receiver: str
    intent: str
    content: str
