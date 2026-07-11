from __future__ import annotations

import argparse

from adaptive_stream_agents.llm.adapters import DeterministicJsonLLM, OllamaChatLLM, OpenAIChatLLM
from adaptive_stream_agents.scenarios.conflict_resolution import (
    build_conflict_resolution_scenario,
    format_conflict_results,
    run_conflict_benchmark,
)
from adaptive_stream_agents.scenarios.traffic_drift import (
    build_traffic_drift_scenario,
    format_traffic_drift_results,
    run_traffic_drift_benchmark,
)
from adaptive_stream_agents.scenarios.prompting_impact import (
    format_prompting_impact_results,
    run_prompting_impact_benchmark,
)
from adaptive_stream_agents.scenarios.initiator_prompting import (
    format_initiator_prompting_results,
    run_initiator_prompting_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MAS-DecStream paper scenarios for LLM-assisted Contract-Net stream scheduling."
    )
    parser.add_argument(
        "--format",
        choices=("text", "latex"),
        default="text",
        help="Output format for the negotiation transcript.",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "conflict",
            "traffic-drift",
            "prompting",
            "initiator-prompting",
        ),
        default="traffic-drift",
        help="Benchmark scenario to run.",
    )
    parser.add_argument(
        "--engine",
        choices=("classic", "langgraph"),
        default="classic",
        help="Run the original Python protocol or the LangGraph agentic workflow.",
    )
    parser.add_argument(
        "--llm",
        choices=("none", "mock", "openai", "ollama"),
        default="mock",
        help="LLM adapter used by agents when running the LangGraph engine.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Default model name for --llm openai or --llm ollama.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for Ollama when using --llm ollama.",
    )
    parser.add_argument(
        "--agent-models",
        default="",
        help=(
            "Optional comma-separated model map, for example "
            "A0=llama3,A1=phi,A2=phi,A3=llama3,A4=ollama/gpt-oss:20b-cloud."
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum number of CFP/refinement rounds before consensus is forced.",
    )
    parser.add_argument(
        "--consensus-margin",
        type=float,
        default=0.15,
        help="Stop when the best proposal exceeds the second-best by this confidence margin.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Stop when the best proposal score changes less than this value between rounds.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.60,
        help="Minimum confidence score required for a proposal to be feasible.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Maximum number of ground-truth rows used by dataset-based scenarios.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="Number of representative edge-cluster agents for scalable scenarios.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=3,
        help="Number of concurrent offloading requests for scalable conflict scenarios.",
    )
    parser.add_argument(
        "--prompt-strategies",
        default="",
        help=(
            "Optional comma-separated prompting strategies for Scenario 3, for example "
            "zero_shot,cot_few_shot,react_few_shot."
        ),
    )
    args = parser.parse_args()

    if args.scenario == "conflict":
        scenario = build_conflict_resolution_scenario(
            num_agents=args.num_agents,
            num_requests=args.num_requests,
        )
        llm_label = _configure_agent_llms(args, scenario.agents)
        print(format_conflict_results(run_conflict_benchmark(scenario, m3_llm_label=llm_label)))
        return

    if args.scenario == "traffic-drift":
        scenario = build_traffic_drift_scenario()
        llm_label = _configure_agent_llms(args, scenario.agents)
        print(
            format_traffic_drift_results(
                run_traffic_drift_benchmark(scenario, m3_llm_label=llm_label)
            )
        )
        return

    if args.scenario == "prompting":
        llm, llm_label = _build_single_llm(args)
        print(
            format_prompting_impact_results(
                run_prompting_impact_benchmark(
                    llm=llm,
                    llm_label=llm_label,
                    limit=args.limit,
                    strategies=_parse_prompt_strategies(args.prompt_strategies),
                )
            )
        )
        return

    if args.scenario == "initiator-prompting":
        llm, llm_label = _build_single_llm(args)
        print(
            format_initiator_prompting_results(
                run_initiator_prompting_benchmark(
                    llm=llm,
                    llm_label=llm_label,
                    limit=args.limit,
                    strategies=_parse_prompt_strategies(args.prompt_strategies),
                    max_rounds=args.max_rounds,
                    consensus_margin=args.consensus_margin,
                    epsilon=args.epsilon,
                )
            )
        )
        return

    raise ValueError(f"Unsupported scenario: {args.scenario}")


def _parse_agent_models(raw_value: str) -> dict[str, str]:
    if not raw_value.strip():
        return {}

    result: dict[str, str] = {}
    for item in raw_value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --agent-models item: {item!r}. Use A1=model_name.")
        agent_id, model = item.split("=", 1)
        result[agent_id.strip()] = model.strip()
    return result


def _configure_agent_llms(args: argparse.Namespace, agents) -> str:
    agent_models = _parse_agent_models(args.agent_models)
    if args.llm in {"none", "mock"}:
        for agent in agents:
            agent.llm = DeterministicJsonLLM()
        return "mock"

    if args.llm == "openai":
        labels: list[str] = []
        for agent in agents:
            model = agent_models.get(agent.agent_id, args.model)
            agent.llm = OpenAIChatLLM(model=model)
            labels.append(f"{agent.agent_id}={model}")
        return "openai(" + ",".join(labels) + ")"

    if args.llm == "ollama":
        labels = []
        for agent in agents:
            model = agent_models.get(agent.agent_id, args.model)
            agent.llm = OllamaChatLLM(model=model, base_url=args.ollama_url)
            labels.append(f"{agent.agent_id}={model}")
        return "ollama(" + ",".join(labels) + ")"

    raise ValueError(f"Unsupported LLM adapter: {args.llm}")


def _build_single_llm(args: argparse.Namespace):
    if args.llm in {"none", "mock"}:
        return None, "mock"
    if args.llm == "openai":
        return OpenAIChatLLM(model=args.model), f"openai({args.model})"
    if args.llm == "ollama":
        return OllamaChatLLM(model=args.model, base_url=args.ollama_url), f"ollama({args.model})"
    raise ValueError(f"Unsupported LLM adapter: {args.llm}")


def _parse_prompt_strategies(raw_value: str):
    if not raw_value.strip():
        return None
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())
