from __future__ import annotations

from typing import Callable, TypedDict


class BenchmarkState(TypedDict):
    context: dict
    proposals: list
    validated: list
    result: object | None


def run_benchmark_graph(
    context: dict,
    propose: Callable[[dict], list],
    validate: Callable[[dict, list], list],
    select: Callable[[dict, list], object],
) -> object:
    """Run a comparable propose-validate-select workflow in LangGraph."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    def proposal_node(state: BenchmarkState) -> BenchmarkState:
        state["proposals"] = propose(state["context"])
        return state

    def validation_node(state: BenchmarkState) -> BenchmarkState:
        state["validated"] = validate(state["context"], state["proposals"])
        return state

    def selection_node(state: BenchmarkState) -> BenchmarkState:
        state["result"] = select(state["context"], state["validated"])
        return state

    workflow = StateGraph(BenchmarkState)
    workflow.add_node("propose", proposal_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("select", selection_node)
    workflow.set_entry_point("propose")
    workflow.add_edge("propose", "validate")
    workflow.add_edge("validate", "select")
    workflow.add_edge("select", END)
    graph = workflow.compile()
    final_state = graph.invoke(
        {
            "context": context,
            "proposals": [],
            "validated": [],
            "result": None,
        }
    )
    return final_state["result"]
