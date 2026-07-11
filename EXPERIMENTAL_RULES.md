# MAS-DecStream Experimental Rules

This file records the fixed assumptions used in the paper experiments.

## Research Questions

**RQ1.** How can agentic AI be leveraged to manage and reduce the complexity of stream-application scheduling under dynamic data rates, heterogeneous resources, and evolving runtime contexts?

**RQ2.** To what extent can LLM-assisted Contract-Net negotiation improve multi-agent coordination for adaptive decision-making in heterogeneous MEC environments?

**RQ3.** How do LLM selection and prompting strategy affect negotiation quality, reasoning latency, token cost, and scheduling robustness?

## Compared Methods

- **RB-SR-CNP:** rule-based multi-agent task offloading using a classic single-round Contract-Net Protocol.
- **RB-MR-CNP:** rule-based multi-agent task offloading using multi-round Contract-Net refinement.
- **LLM-MR-CNP:** proposed approach using LLM-assisted reasoning and multi-round Contract-Net negotiation.

All methods use agents and communication. Only **LLM-MR-CNP** uses LLM reasoning.

## Final Decision Rule

The final offloading host must not be selected from LLM confidence alone. Candidate proposals are first filtered using resource, QoS, and contextual constraints. The final host is then selected by maximizing the objective-based utility among feasible candidates.

## Paper Scenarios

### Scenario 1: Adaptive Stream-Task Migration Under Data Drift

This scenario addresses **RQ1** and evaluates robustness when an unforeseen workload spike is missed by the quantitative prediction model.

Command:

```powershell
python -m adaptive_stream_agents --scenario traffic-drift --llm ollama --model chevalblanc/gpt-4o-mini
```

### Scenario 2: Conflict Resolution Under Concurrent Offloading Requests

This scenario addresses **RQ1** and **RQ2** by increasing concurrent offloading requests and the number of representative agents.

Commands:

```powershell
python -m adaptive_stream_agents --scenario conflict --llm ollama --model chevalblanc/gpt-4o-mini --num-agents 5 --num-requests 3
python -m adaptive_stream_agents --scenario conflict --llm ollama --model chevalblanc/gpt-4o-mini --num-agents 10 --num-requests 6
python -m adaptive_stream_agents --scenario conflict --llm ollama --model chevalblanc/gpt-4o-mini --num-agents 20 --num-requests 12
```

### Scenario 3: LLM Reasoning and Prompting Strategy Evaluation

This scenario addresses **RQ3** by evaluating LLM capability, prompting strategy, CFP formulation accuracy, responder action accuracy, final offloading accuracy, decision time, and token cost.

Responder-only prompting, kept as Experiment 3.2-A:

```powershell
python -m adaptive_stream_agents --scenario prompting --llm ollama --model deepseek-v4-pro:cloud --limit 25
```

End-to-end initiator/responder negotiation, renamed Experiment 3.2-B:

```powershell
python -m adaptive_stream_agents --scenario initiator-prompting --llm ollama --model "gemini-3-flash-preview:cloud" --limit 25 --prompt-strategies "zero_shot" --max-rounds 3
```

## LLM Reporting Rule

Every LLM-based experiment must report:

- model name;
- local or cloud execution mode;
- prompting strategy;
- decision time;
- token cost;
- final scheduling quality.

## Guardrails

- Do not claim that the LLM directly optimizes the objective function.
- Do not select a host only from LLM confidence.
- Do not compare methods using different task data within the same scenario.
- Always report the operational cost of LLM reasoning.
