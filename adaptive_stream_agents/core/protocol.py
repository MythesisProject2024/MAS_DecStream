from __future__ import annotations

from dataclasses import dataclass

from adaptive_stream_agents.core.agent import EdgeClusterAgent
from adaptive_stream_agents.core.domain import NegotiationMessage, Proposal, StreamTask
from adaptive_stream_agents.core.transcript import Transcript


@dataclass(frozen=True)
class ConsensusDecision:
    selected_agent_id: str | None
    confidence: float
    reason: str
    migration_approved: bool = True


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str


class ContractNetSession:
    """Variant of Contract-Net Protocol with natural-language rounds."""

    def __init__(
        self,
        agents: list[EdgeClusterAgent],
        task: StreamTask,
        initiator_id: str,
        max_rounds: int = 3,
        consensus_margin: float = 0.15,
        epsilon: float = 0.03,
        min_score: float = 0.60,
    ) -> None:
        self.agents = {agent.agent_id: agent for agent in agents}
        self.task = task
        self.initiator_id = initiator_id
        self.max_rounds = max_rounds
        self.consensus_margin = consensus_margin
        self.epsilon = epsilon
        self.min_score = min_score
        self.transcript = Transcript()

    def run(self) -> Transcript:
        candidate_agents = [agent for agent in self.agents.values() if agent.agent_id != self.initiator_id]

        self._broadcast(
            round_id=1,
            intent="Call for Proposals",
            content=f"{self.task.stream_type} stream workload exceeds local capacity. CFP: request migration of task {self.task.task_id}.",
            receivers=[agent.agent_id for agent in candidate_agents],
        )
        proposals_by_round: list[list[Proposal]] = [
            self._collect_proposals(candidate_agents, round_id=1)
        ]

        stop = self._evaluate_stopping_conditions(proposals_by_round, round_id=1)
        current_round = 1
        while not stop.should_stop and current_round < self.max_rounds:
            current_round += 1
            viable = self._feasible_proposals(proposals_by_round[-1])
            refinement = (
                "Workload is dynamic; refine proposal using near-future load, QoS, stability, and resource risk."
            )
            self._broadcast(
                round_id=current_round,
                intent="Prediction-Aware Refinement",
                content=(
                    "Current proposals are still close or unstable. Please refine considering "
                    "near-future evolution and whether your confidence remains above the minimum score."
                ),
                receivers=[proposal.agent_id for proposal in viable],
            )
            proposals_by_round.append(
                self._collect_proposals(
                    [self.agents[proposal.agent_id] for proposal in viable],
                    round_id=current_round,
                    refinement=refinement,
                )
            )
            stop = self._evaluate_stopping_conditions(proposals_by_round, round_id=current_round)

        if current_round >= self.max_rounds and not stop.should_stop:
            stop = StopDecision(True, f"Maximum refinement rounds reached ({self.max_rounds}).")

        decision = self._select_consensus(proposals_by_round[-1], stop.reason)
        self._broadcast_consensus(decision, round_id=current_round + 1)
        return self.transcript

    def _broadcast(self, round_id: int, intent: str, content: str, receivers: list[str]) -> None:
        for receiver in receivers:
            self.transcript.add(
                NegotiationMessage(
                    round_id=round_id,
                    sender=self.initiator_id,
                    receiver=receiver,
                    intent=intent,
                    content=content,
                )
            )

    def _collect_proposals(
        self,
        agents: list[EdgeClusterAgent],
        round_id: int,
        refinement: str | None = None,
    ) -> list[Proposal]:
        proposals: list[Proposal] = []
        for agent in agents:
            proposal = agent.evaluate_cfp(self.task, round_id=round_id, refinement=refinement)
            proposals.append(proposal)
            self.transcript.add(
                NegotiationMessage(
                    round_id=round_id,
                    sender=agent.agent_id,
                    receiver=self.initiator_id,
                    intent="Proposal" if proposal.accepted else "Refusal",
                    content=proposal.natural_language,
                )
            )
        return proposals

    def _evaluate_stopping_conditions(
        self,
        proposals_by_round: list[list[Proposal]],
        round_id: int,
    ) -> StopDecision:
        current = proposals_by_round[-1]
        feasible = self._feasible_proposals(current)

        if not feasible:
            return StopDecision(True, "No feasible candidate satisfies the minimum score.")

        ranked = sorted(feasible, key=lambda p: (p.confidence, p.utility), reverse=True)
        best = ranked[0]
        if len(ranked) >= 2:
            second_best = ranked[1]
            margin = best.confidence - second_best.confidence
            if margin >= self.consensus_margin:
                return StopDecision(
                    True,
                    (
                        f"Consensus margin reached: {best.agent_id} exceeds "
                        f"{second_best.agent_id} by {margin:.2f}."
                    ),
                )

        if round_id >= self.max_rounds:
            return StopDecision(True, f"Maximum refinement rounds reached ({self.max_rounds}).")

        if len(proposals_by_round) >= 2 and best.confidence >= self.min_score:
            previous_best = max(
                self._feasible_proposals(proposals_by_round[-2]),
                key=lambda p: (p.confidence, p.utility),
                default=None,
            )
            if previous_best is not None:
                delta = abs(best.confidence - previous_best.confidence)
                if delta < self.epsilon:
                    return StopDecision(
                        True,
                        (
                            f"Proposal scores converged: best confidence changed by {delta:.2f}, "
                            f"below epsilon {self.epsilon:.2f}."
                        ),
                    )

        return StopDecision(False, "Refinement should continue.")

    def _feasible_proposals(self, proposals: list[Proposal]) -> list[Proposal]:
        return [
            proposal
            for proposal in proposals
            if proposal.accepted and proposal.confidence >= self.min_score
        ]

    def _select_consensus(self, proposals: list[Proposal], stop_reason: str) -> ConsensusDecision:
        accepted = self._feasible_proposals(proposals)
        if not accepted:
            return ConsensusDecision(
                selected_agent_id=None,
                confidence=0.0,
                reason=f"{stop_reason} Final decision: keep the task local or offload to cloud.",
                migration_approved=False,
            )

        best = max(accepted, key=lambda p: (p.confidence, p.utility))
        return ConsensusDecision(
            selected_agent_id=best.agent_id,
            confidence=best.confidence,
            reason=f"{stop_reason} Agent {best.agent_id} provides the most stable execution context.",
        )

    def _broadcast_consensus(self, decision: ConsensusDecision, round_id: int) -> None:
        if decision.migration_approved:
            content = (
                f"{decision.reason} Final decision: migrate {self.task.stream_type} task "
                f"{self.task.task_id} to {decision.selected_agent_id} with confidence {decision.confidence:.2f}."
            )
        else:
            content = (
                f"{decision.reason} No edge-cluster migration is approved for "
                f"{self.task.stream_type} task {self.task.task_id}."
            )
        self._broadcast(
            round_id=round_id,
            intent="Consensus Decision",
            content=content,
            receivers=[agent_id for agent_id in self.agents if agent_id != self.initiator_id],
        )
        for agent_id, agent in self.agents.items():
            if agent_id == self.initiator_id:
                continue
            self.transcript.add(
                NegotiationMessage(
                    round_id=round_id,
                    sender=agent_id,
                    receiver=self.initiator_id,
                    intent="Acknowledgement",
                    content=(
                        agent.acknowledge_decision(decision.selected_agent_id, self.task)
                        if decision.selected_agent_id is not None
                        else "Acknowledged. No feasible edge migration is selected."
                    ),
                )
            )
