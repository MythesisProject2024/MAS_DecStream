from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from adaptive_stream_agents.core.agent import EdgeClusterAgent
from adaptive_stream_agents.core.domain import NegotiationMessage, Proposal, StreamTask
from adaptive_stream_agents.core.protocol import ConsensusDecision, StopDecision
from adaptive_stream_agents.core.transcript import Transcript


class NegotiationState(TypedDict):
    round_id: int
    proposals_by_round: list[list[Proposal]]
    active_agent_ids: list[str]
    stop: StopDecision | None
    decision: ConsensusDecision | None
    transcript: Transcript


@dataclass(frozen=True)
class ContractNetConfig:
    max_rounds: int = 3
    consensus_margin: float = 0.15
    epsilon: float = 0.03
    min_score: float = 0.60


class LangGraphContractNetSession:
    """Contract-Net variant implemented as a LangGraph state graph."""

    def __init__(
        self,
        agents: list[EdgeClusterAgent],
        task: StreamTask,
        initiator_id: str,
        config: ContractNetConfig | None = None,
    ) -> None:
        self.agents = {agent.agent_id: agent for agent in agents}
        self.task = task
        self.initiator_id = initiator_id
        self.config = config or ContractNetConfig()

    def run(self) -> Transcript:
        graph = self._compile_graph()
        candidate_ids = [agent_id for agent_id in self.agents if agent_id != self.initiator_id]
        initial_state: NegotiationState = {
            "round_id": 0,
            "proposals_by_round": [],
            "active_agent_ids": candidate_ids,
            "stop": None,
            "decision": None,
            "transcript": Transcript(),
        }
        final_state = graph.invoke(initial_state)
        return final_state["transcript"]

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph is not installed. Install dependencies with: pip install -r requirements.txt"
            ) from exc

        workflow = StateGraph(NegotiationState)
        workflow.add_node("cfp", self._cfp_node)
        workflow.add_node("evaluate_stop", self._evaluate_stop_node)
        workflow.add_node("refinement", self._refinement_node)
        workflow.add_node("consensus", self._consensus_node)

        workflow.set_entry_point("cfp")
        workflow.add_edge("cfp", "evaluate_stop")
        workflow.add_conditional_edges(
            "evaluate_stop",
            self._route_after_stop_check,
            {"refinement": "refinement", "consensus": "consensus"},
        )
        workflow.add_edge("refinement", "evaluate_stop")
        workflow.add_edge("consensus", END)
        return workflow.compile()

    def _cfp_node(self, state: NegotiationState) -> NegotiationState:
        round_id = 1
        transcript = state["transcript"]
        self._broadcast(
            transcript,
            round_id=round_id,
            intent="Call for Proposals",
            content=(
                f"{self.task.stream_type} stream workload exceeds local capacity. "
                f"CFP: request migration of task {self.task.task_id}."
            ),
            receivers=state["active_agent_ids"],
        )
        proposals = self._collect_proposals(state["active_agent_ids"], round_id)
        transcript.extend(self._proposal_messages(proposals, round_id))
        state["round_id"] = round_id
        state["proposals_by_round"] = [proposals]
        return state

    def _evaluate_stop_node(self, state: NegotiationState) -> NegotiationState:
        state["stop"] = self._evaluate_stopping_conditions(
            state["proposals_by_round"], state["round_id"]
        )
        return state

    def _refinement_node(self, state: NegotiationState) -> NegotiationState:
        round_id = state["round_id"] + 1
        feasible = self._feasible_proposals(state["proposals_by_round"][-1])
        active_agent_ids = [proposal.agent_id for proposal in feasible]
        transcript = state["transcript"]
        refinement = (
            "Workload is dynamic; refine proposal using near-future load, QoS, stability, "
            "and resource risk."
        )

        self._broadcast(
            transcript,
            round_id=round_id,
            intent="Prediction-Aware Refinement",
            content=(
                "Current proposals are still close or unstable. Please refine considering "
                "near-future evolution and whether your confidence remains above the minimum score."
            ),
            receivers=active_agent_ids,
        )
        proposals = self._collect_proposals(active_agent_ids, round_id, refinement)
        transcript.extend(self._proposal_messages(proposals, round_id))

        state["round_id"] = round_id
        state["active_agent_ids"] = active_agent_ids
        state["proposals_by_round"] = state["proposals_by_round"] + [proposals]
        return state

    def _consensus_node(self, state: NegotiationState) -> NegotiationState:
        stop = state["stop"] or StopDecision(True, "Negotiation stopped.")
        decision = self._select_consensus(state["proposals_by_round"][-1], stop.reason)
        state["decision"] = decision
        self._broadcast_consensus(state["transcript"], decision, round_id=state["round_id"] + 1)
        return state

    def _route_after_stop_check(self, state: NegotiationState) -> str:
        stop = state["stop"]
        if stop is not None and stop.should_stop:
            return "consensus"
        return "refinement"

    def _broadcast(
        self,
        transcript: Transcript,
        round_id: int,
        intent: str,
        content: str,
        receivers: list[str],
    ) -> None:
        for receiver in receivers:
            transcript.add(
                NegotiationMessage(
                    round_id=round_id,
                    sender=self.initiator_id,
                    receiver=receiver,
                    intent=intent,
                    content=content,
                )
            )

    def _collect_proposals(
        self, agent_ids: list[str], round_id: int, refinement: str | None = None
    ) -> list[Proposal]:
        return [
            self.agents[agent_id].evaluate_cfp(self.task, round_id=round_id, refinement=refinement)
            for agent_id in agent_ids
        ]

    def _proposal_messages(self, proposals: list[Proposal], round_id: int) -> list[NegotiationMessage]:
        return [
            NegotiationMessage(
                round_id=round_id,
                sender=proposal.agent_id,
                receiver=self.initiator_id,
                intent="Proposal" if proposal.accepted else "Refusal",
                content=proposal.natural_language,
            )
            for proposal in proposals
        ]

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
            if margin >= self.config.consensus_margin:
                return StopDecision(
                    True,
                    (
                        f"Consensus margin reached: {best.agent_id} exceeds "
                        f"{second_best.agent_id} by {margin:.2f}."
                    ),
                )

        if round_id >= self.config.max_rounds:
            return StopDecision(True, f"Maximum refinement rounds reached ({self.config.max_rounds}).")

        if len(proposals_by_round) >= 2 and best.confidence >= self.config.min_score:
            previous_best = max(
                self._feasible_proposals(proposals_by_round[-2]),
                key=lambda p: (p.confidence, p.utility),
                default=None,
            )
            if previous_best is not None:
                delta = abs(best.confidence - previous_best.confidence)
                if delta < self.config.epsilon:
                    return StopDecision(
                        True,
                        (
                            f"Proposal scores converged: best confidence changed by {delta:.2f}, "
                            f"below epsilon {self.config.epsilon:.2f}."
                        ),
                    )

        return StopDecision(False, "Refinement should continue.")

    def _feasible_proposals(self, proposals: list[Proposal]) -> list[Proposal]:
        return [
            proposal
            for proposal in proposals
            if proposal.accepted and proposal.confidence >= self.config.min_score
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

    def _broadcast_consensus(
        self, transcript: Transcript, decision: ConsensusDecision, round_id: int
    ) -> None:
        if decision.migration_approved:
            content = (
                f"{decision.reason} Final decision: migrate {self.task.stream_type} task "
                f"{self.task.task_id} to {decision.selected_agent_id} with confidence "
                f"{decision.confidence:.2f}."
            )
        else:
            content = (
                f"{decision.reason} No edge-cluster migration is approved for "
                f"{self.task.stream_type} task {self.task.task_id}."
            )

        self._broadcast(
            transcript,
            round_id=round_id,
            intent="Consensus Decision",
            content=content,
            receivers=[agent_id for agent_id in self.agents if agent_id != self.initiator_id],
        )
        for agent_id, agent in self.agents.items():
            if agent_id == self.initiator_id:
                continue
            transcript.add(
                NegotiationMessage(
                    round_id=round_id,
                    sender=agent_id,
                    receiver=self.initiator_id,
                    intent="Acknowledgement",
                    content=agent.acknowledge_decision(decision.selected_agent_id, self.task),
                )
            )
