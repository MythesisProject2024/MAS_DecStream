# MAS-DecStream

MAS-DecStream is a  agentic-AI framework for decentralized stream-task scheduling in heterogeneous MEC environments. It  implements LLM-assisted Contract-Net negotiation between representative edge-cluster agents.

The clean project contains only the scenarios used in the paper.

## Installation

```powershell
pip install -r requirements.txt
```

Ollama models can be local lightweight models or Ollama cloud-routed models.

## Paper Scenarios

### Scenario 1: Adaptive Stream-Task Migration Under Data Drift

This scenario evaluates robustness under dynamic workloads and unforeseen runtime conditions. An unexpected workload spike is missed by the quantitative prediction model, then the initiator agent formulates a CFP and starts multi-round offloading negotiation.

```powershell
python -m adaptive_stream_agents --scenario traffic-drift --llm mock
python -m adaptive_stream_agents --scenario traffic-drift --llm ollama --model chevalblanc/gpt-4o-mini
```

### Scenario 2: Conflict Resolution Under Concurrent Offloading Requests

This scenario evaluates conflict resolution and scalability under concurrent offloading requests. It supports increasing the number of representative agents and simultaneous requests.

```powershell
python -m adaptive_stream_agents --scenario conflict --llm mock --num-agents 5 --num-requests 3
python -m adaptive_stream_agents --scenario conflict --llm ollama --model chevalblanc/gpt-4o-mini --num-agents 5 --num-requests 3
python -m adaptive_stream_agents --scenario conflict --llm ollama --model chevalblanc/gpt-4o-mini --num-agents 10 --num-requests 6
python -m adaptive_stream_agents --scenario conflict --llm ollama --model chevalblanc/gpt-4o-mini --num-agents 20 --num-requests 12
```

### Scenario 3: LLM Reasoning and Prompting Strategy Evaluation

Scenario 3 studies how LLM capability and prompting strategy affect negotiation quality, reasoning latency, token cost, and final offloading accuracy.

#### Experiment 3.1: Ablation Study on LLM Capability and Negotiation Strategy

Use the same scenario with different LLMs and negotiation depths. A one-shot run can be approximated with `--max-rounds 1`, while the proposed multi-round setting uses `--max-rounds 3`.

```powershell
python -m adaptive_stream_agents --scenario initiator-prompting --llm ollama --model phi3 --limit 25 --prompt-strategies zero_shot --max-rounds 1
python -m adaptive_stream_agents --scenario initiator-prompting --llm ollama --model gpt-oss:20b:cloud --limit 25 --prompt-strategies zero_shot --max-rounds 3
```

#### Experiment 3.2-A: Responder-Agent Prompting Evaluation

This is the older responder-only prompting experiment. It evaluates whether responder agents correctly generate accept/refuse decisions under different prompting strategies, without full end-to-end initiator negotiation.

```powershell
python -m adaptive_stream_agents --scenario prompting --llm ollama --model deepseek-v4-pro:cloud --limit 25
```

#### Experiment 3.2-B: Initiator-Agent Reasoning and Final Consensus Evaluation

This is the end-to-end LLM-MR-CNP negotiation experiment. The initiator formulates the CFP, responders propose or refuse, refinement may occur, and the final host is selected by feasibility validation and objective-based utility.

```powershell
python -m adaptive_stream_agents --scenario initiator-prompting --llm ollama --model "gemini-3-flash-preview:cloud" --limit 25 --prompt-strategies "zero_shot" --max-rounds 3
python -m adaptive_stream_agents --scenario initiator-prompting --llm ollama --model "gpt-oss:20b:cloud" --limit 25 --prompt-strategies "zero_shot,few_shot,cot,cot_few_shot,react,react_few_shot_cot" --max-rounds 3
```

## Method Names

- `RB-SR-CNP`: rule-based multi-agent single-round Contract-Net.
- `RB-MR-CNP`: rule-based multi-agent multi-round Contract-Net with refinement.
- `LLM-MR-CNP`: proposed LangGraph/LLM-assisted multi-round Contract-Net negotiation.

Only `LLM-MR-CNP` uses LLM reasoning. All methods use agents and Contract-Net communication.

## Decision Rule

The final host is not selected using LLM confidence alone. The decision process is:

1. Apply hard resource and QoS feasibility constraints.
2. Validate contextual constraints such as drift, priority, or reliability warnings.
3. Rank feasible candidates using objective-based utility.
4. Select the valid candidate with the highest utility.

## Project Structure

- `adaptive_stream_agents/core`: domain models and Contract-Net protocol utilities.
- `adaptive_stream_agents/llm`: mock, OpenAI, and Ollama LLM adapters.
- `adaptive_stream_agents/scenarios/traffic_drift.py`: Scenario 1.
- `adaptive_stream_agents/scenarios/conflict_resolution.py`: Scenario 2.
- `adaptive_stream_agents/scenarios/prompting_impact.py`: Scenario 3.2-A responder-only prompting.
- `adaptive_stream_agents/scenarios/initiator_prompting.py`: Scenario 3.2-B end-to-end negotiation.
- `config`: prompt templates.
- `datasets`: ground-truth files.
- `outputs`: saved experiment outputs.
