from __future__ import annotations

import json
from typing import Any

from adaptive_stream_agents.core.domain import ClusterState, Proposal, StreamTask
from adaptive_stream_agents.llm.adapters import LLMAdapter


class EdgeClusterAgent:
    """Representative edge node with local cluster visibility."""

    def __init__(
        self,
        agent_id: str,
        state: ClusterState,
        llm: LLMAdapter | None = None,
        qualitative_context: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.state = state
        self.llm = llm
        self.qualitative_context = qualitative_context

    def evaluate_cfp(self, task: StreamTask, round_id: int, refinement: str | None = None) -> Proposal:
        if self.llm is not None:
            return self._evaluate_with_llm(task, round_id, refinement)
        return self._evaluate_deterministically(task, round_id, refinement)

    def _evaluate_deterministically(
        self, task: StreamTask, round_id: int, refinement: str | None = None
    ) -> Proposal:
        score = self._score(task, prediction_aware=round_id >= 2)
        feasible = (
            self.state.cpu_available >= task.cpu_demand
            and self.state.memory_available >= task.memory_demand
            and self.state.bandwidth_available >= task.bandwidth_demand
            and self.state.avg_latency_ms <= task.latency_budget_ms
        )
        accepted = feasible and score >= 0.52
        confidence = max(0.0, min(score, 1.0))
        rationale = self._rationale(task, confidence, round_id, refinement)
        return Proposal(
            agent_id=self.agent_id,
            accepted=accepted,
            confidence=confidence,
            utility=round(confidence * 100, 2),
            rationale=rationale,
            natural_language=self._natural_language(task, accepted, confidence, round_id, rationale),
        )

    def _evaluate_with_llm(
        self, task: StreamTask, round_id: int, refinement: str | None = None
    ) -> Proposal:
        deterministic = self._evaluate_deterministically(task, round_id, refinement)
        system_prompt = (
            "You are a representative edge-cluster scheduling agent. "
            "You reason only from your local cluster state. "
            "Reply with valid JSON using keys: accepted, confidence, rationale, natural_language. "
            "confidence must be a number between 0 and 1."
        )
        user_prompt = json.dumps(
            {
                "agent_id": self.agent_id,
                "round_id": round_id,
                "task": task.__dict__,
                "local_cluster_state": self.state.__dict__,
                "qualitative_context": self.qualitative_context,
                "refinement_request": refinement,
                "deterministic_baseline": {
                    "accepted": deterministic.accepted,
                    "confidence": deterministic.confidence,
                    "rationale": deterministic.rationale,
                },
            },
            indent=2,
        )

        raw_response = self.llm.complete(system_prompt, user_prompt)
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return Proposal(
                agent_id=self.agent_id,
                accepted=deterministic.accepted,
                confidence=deterministic.confidence,
                utility=deterministic.utility,
                rationale=deterministic.rationale,
                natural_language=raw_response.strip() or deterministic.natural_language,
            )

        accepted = bool(parsed.get("accepted", deterministic.accepted))
        confidence = self._as_confidence(parsed.get("confidence", deterministic.confidence))
        rationale = str(parsed.get("rationale", deterministic.rationale))
        natural_language = str(parsed.get("natural_language", deterministic.natural_language))
        return Proposal(
            agent_id=self.agent_id,
            accepted=accepted,
            confidence=confidence,
            utility=round(confidence * 100, 2),
            rationale=rationale,
            natural_language=natural_language,
        )

    def _as_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0
        return round(max(0.0, min(confidence, 1.0)), 3)

    def acknowledge_decision(self, selected_agent_id: str | None, task: StreamTask) -> str:
        if selected_agent_id is None:
            return "Acknowledged. No feasible edge migration is selected."
        if selected_agent_id == self.agent_id:
            return f"Acknowledged. Task migration accepted for {task.stream_type} stream {task.task_id}."
        return "Acknowledged. I defer participation and keep local resources available."

    def _score(self, task: StreamTask, prediction_aware: bool) -> float:
        state = self.state
        cpu_margin = (state.cpu_available - task.cpu_demand) / max(state.cpu_capacity, 1.0)
        mem_margin = (state.memory_available - task.memory_demand) / max(state.memory_capacity, 1.0)
        bw_margin = (state.bandwidth_available - task.bandwidth_demand) / max(state.bandwidth_capacity, 1.0)
        latency_fit = 1.0 if state.avg_latency_ms <= task.latency_budget_ms else 0.35

        future_penalty = 0.0
        if prediction_aware:
            future_penalty = 0.35 * state.predicted_cpu_growth + 0.20 * state.predicted_memory_growth

        energy_penalty = 0.15 * state.energy_pressure
        raw = (
            0.30 * cpu_margin
            + 0.20 * mem_margin
            + 0.15 * bw_margin
            + 0.20 * state.stability_score
            + 0.15 * latency_fit
            - future_penalty
            - energy_penalty
        )
        return round(max(0.0, min(1.0, raw + 0.45)), 3)

    def _rationale(
        self, task: StreamTask, confidence: float, round_id: int, refinement: str | None
    ) -> str:
        state = self.state
        reasons: list[str] = []
        if state.cpu_available >= task.cpu_demand:
            reasons.append("CPU margin is sufficient")
        else:
            reasons.append("CPU margin is tight")
        if state.avg_latency_ms <= task.latency_budget_ms:
            reasons.append("latency remains within budget")
        else:
            reasons.append("latency risk is above budget")
        if round_id >= 2:
            if state.predicted_cpu_growth + state.predicted_memory_growth > 0.55:
                reasons.append("future workload growth may reduce QoS")
            else:
                reasons.append("near-future load prediction is stable")
        if refinement:
            reasons.append(f"refinement considered: {refinement}")
        reasons.append(f"confidence={confidence:.2f}")
        return "; ".join(reasons)

    def _natural_language(
        self, task: StreamTask, accepted: bool, confidence: float, round_id: int, rationale: str
    ) -> str:
        if accepted and round_id == 1:
            opening = f"I can accommodate the {task.stream_type} workload and am willing to host it."
        elif accepted:
            opening = "Continued acceptance confirmed after prediction-aware evaluation."
        elif round_id >= 2:
            opening = "Expected workload evolution may impact QoS, so my acceptance confidence is reduced."
        else:
            opening = "The task is risky under current local conditions, so I decline participation."
        return f"{opening} Local reasoning: {rationale}."
