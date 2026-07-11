# Alibaba Job Summary Preprocessing

This file describes the preprocessing pipeline applied to the real Alibaba ASI job execution summary sample.

## Input

The input is a real Alibaba trace sample:

```text
data/asi_job_execution_summary_sample_1000.csv
```

It was extracted from:

```text
data/asi_opensource_job_execution_summary.zip
```

## Step 1: Clean Real Workload Features

The script keeps relevant workload columns:

- `pod_id`
- `workload_id`
- `server_id`
- `gpu_spec_public`
- `priority_class`
- `job_type_public`
- `model_type_public`
- `is_genai_request`
- `gpu_request`
- `duration_hours`
- `schedule_delay_sec`
- `ready_delay_sec`
- `ready_status`
- `schedule_status`

It normalizes selected numerical values and derives:

- `scenario_family`
- `stream_task_type`
- `semantic_requirements`
- `latency_deadline_ms`
- `bandwidth_request_norm`
- `qualitative_warnings`
- `natural_language_request`

Output:

```text
datasets/alibaba_job_summary_preprocessed.csv
```

## Step 2: Workload-Type Mapping

Workloads are mapped to the scenario families used in A2-Stream:

- `dynamic_workload`: long-running or high-priority online workloads.
- `data_drift`: offline or low-priority workloads that may hide delayed bursts.
- `unseen_task`: workloads with model types such as `cv`, `rec`, or `embedding`.
- `conflict_resolution`: workloads with high scheduling delay or readiness failure, indicating contention.

This mapping is heuristic and should be reported as an augmentation rule, not as an original Alibaba label.

## Step 3: MEC Augmentation

The real Alibaba workload rows are augmented with edge/MEC scheduling fields:

- natural-language offloading request;
- latency deadline;
- bandwidth demand;
- semantic requirements;
- qualitative warnings;
- candidate edge-cluster snapshots;
- feasibility labels;
- preferred cluster according to normalized utility.

Output:

```text
datasets/alibaba_job_summary_augmented.json
datasets/alibaba_job_summary_augmented_workloads.csv
datasets/alibaba_job_summary_augmented_cluster_snapshots.csv
datasets/alibaba_job_summary_augmented_ground_truth.csv
```

## Current Generated Sample

The current generated dataset uses 1000 real Alibaba rows and produces:

- 1000 workload records;
- 5000 edge-cluster snapshots;
- 1000 ground-truth placement labels.

The sample is not artificially balanced. Its scenario-family distribution reflects the selected real trace sample and the heuristic mapping rules.

## Reproducibility

Run:

```powershell
python tools\preprocess_alibaba_sample.py --rows 1000
```

For 2000 or 5000 rows, first extract a larger CSV sample from the ZIP, then run:

```powershell
python tools\preprocess_alibaba_sample.py --input data\asi_job_execution_summary_sample_5000.csv --rows 5000
```
