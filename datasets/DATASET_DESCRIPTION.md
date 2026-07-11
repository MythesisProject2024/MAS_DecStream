# A2-Stream Alibaba-Augmented Dataset Description

This document describes the workload dataset used to evaluate A2-Stream and the compared multi-agent scheduling methods.

## Important Principle

The dataset is **independent of the scheduling technique**. It does not define agents, LangGraph nodes, or Contract-Net behavior. It defines:

- stream-processing workloads that must be deployed;
- edge-cluster resource snapshots;
- evaluation labels for feasible and preferred placements.

In the experimental framework, each edge cluster may later be represented by a cooperative scheduling agent. This mapping belongs to the method implementation, not to the dataset itself.

## Provenance

The current dataset is an **Alibaba-inspired augmented dataset**. It imitates normalized features commonly derived from cluster workload traces, such as CPU request, memory request, runtime usage, instance count, queue delay, execution duration, and workload variation.

It is augmented with MEC and stream-scheduling features:

- latency deadlines;
- bandwidth demand;
- predicted data-rate evolution;
- predicted resource usage horizon;
- semantic requirements;
- qualitative runtime warnings;
- natural-language scheduling requests.

Scientific wording:

- Use **Alibaba-inspired augmented dataset** for the current generated version.
- Use **Alibaba-augmented dataset** only after real Alibaba trace rows are ingested, normalized, filtered, and augmented by this pipeline.

## Research Questions Supported

- **RQ1:** How can agentic AI reduce the complexity of stream-application scheduling under dynamic data rates, heterogeneous resources, and evolving runtime contexts?
- **RQ2:** To what extent can LLM-assisted Contract-Net negotiation improve multi-agent coordination for adaptive decision-making in heterogeneous MEC environments?
- **RQ3:** How do LLM selection and prompting strategy affect negotiation quality, reasoning latency, token cost, and scheduling robustness in LLM-assisted multi-agent offloading?

## Dataset Size And Families

The dataset contains **100 workload-placement records** divided into four balanced families:

| Family | Records | Purpose |
|---|---:|---|
| `dynamic_workload` | 25 | Evaluate scheduling under fluctuating data rates and predicted workload evolution. |
| `data_drift` | 25 | Evaluate robustness to abnormal spikes and qualitative warnings not fully captured by numerical forecasts. |
| `unseen_task` | 25 | Evaluate adaptation to unseen stream-processing tasks and semantic constraints. |
| `conflict_resolution` | 25 | Evaluate placement under simultaneous workload contention and shared-resource pressure. |

## JSON Structure

Each record contains:

- `record_id`: unique dataset record identifier.
- `workload`: stream task and normalized workload features.
- `edge_cluster_snapshots`: resource states of candidate edge clusters.
- `evaluation_labels`: feasible clusters, unsafe clusters, preferred cluster, and utility labels.

## Workload Fields

Each workload includes:

- workload identifier;
- scenario family;
- natural-language scheduling request;
- application name and domain;
- stream task type;
- criticality and dynamicity;
- latency deadline;
- bandwidth demand;
- semantic requirements;
- qualitative warnings;
- normalized Alibaba-like trace features.

The Alibaba-like normalized trace features include:

- `cpu_request_norm`;
- `memory_request_norm`;
- `gpu_request_norm`;
- `instance_count_norm`;
- `queue_delay_norm`;
- `execution_duration_norm`;
- `mean_cpu_usage_norm`;
- `peak_cpu_usage_norm`;
- `mean_memory_usage_norm`;
- `peak_memory_usage_norm`;
- `data_rate_norm_t`;
- `predicted_data_rate_norm`;
- `predicted_cpu_usage_norm`;
- `observed_unmodelled_spike_norm`.

## Edge-Cluster Snapshot Fields

Each edge-cluster snapshot includes:

- cluster identifier;
- cluster type;
- number of computing nodes;
- normalized CPU, memory, and bandwidth capacity;
- current CPU, memory, and bandwidth usage;
- current latency;
- energy efficiency;
- trust score;
- privacy-enclave support;
- low-latency-tier support;
- predicted CPU and memory usage horizon;
- qualitative warnings.

## Evaluation Labels

The evaluation labels are used only for benchmarking. They are not directly shown to scheduling methods.

They include:

- feasible clusters;
- unsafe clusters;
- preferred cluster;
- cluster utility values;
- label explanation.

The preferred cluster is the feasible cluster with the highest normalized utility after hard resource, latency, semantic, and contextual validation.

## CSV Files

The dataset is exported into three CSV files:

```text
datasets/a2_stream_workloads.csv
datasets/a2_stream_cluster_snapshots.csv
datasets/a2_stream_ground_truth.csv
```

The JSON source is:

```text
datasets/a2_stream_alibaba_augmented_100.json
```

The deterministic generation script is:

```text
tools/generate_dataset.py
```

The CSV export script is:

```text
tools/export_dataset_csv.py
```
