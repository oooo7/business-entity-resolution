# Business Entity Resolution — Competition Project

## Project Overview
This repository contains an end-to-end Machine Learning solution for the **Business Entity Resolution Challenge**. Given business entity records from 3 independent, noisy data sources (`Source 1`, `Source 2`, `Source 3`), the goal is to resolve and match records corresponding to the same real-world business entity.

- **Current Development Phase**: Phase 1 — Data Loading, Profiling, Ground Truth Analysis, Validation Split & Metric Infrastructure.
- **Strict Constraint**: No external business databases, commercial APIs, geocoding APIs, or data augmentation are used. Everything relies purely on the provided training dataset.

---

## Directory Structure
```
.
├── dataset -> student_resource/dataset # Symlink to dataset directory
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── student_resource/                   # Original competition student resources
├── src/                                # Core source code library
│   ├── __init__.py
│   ├── data/                           # Data loading, profiling & split modules
│   │   ├── __init__.py
│   │   ├── loader.py                   # Reusable TSV loader and schema validator
│   │   └── validation_split.py         # Entity-level 80/20 stratified validation split
│   └── evaluation/                     # Metric evaluation infrastructure
│       ├── __init__.py
│       ├── metrics.py                  # Macro F0.5 & country-level metric evaluator
│       └── measure_blocking_recall.py  # Blocking recall & channel evaluator
├── experiments/                        # Experiment tracking and reports
│   ├── experiment_log.csv              # Experiment log
│   ├── EDA_REPORT.md                   # Comprehensive 18-section EDA report
│   ├── validation_split_info.json      # Validation split statistics
│   ├── train_s1_ids.json               # Train split entity IDs
│   └── val_s1_ids.json                 # Validation split entity IDs
├── tests/                              # Comprehensive unit test suite
│   ├── test_loader.py
│   ├── test_metrics.py
│   └── test_blocking_recall.py
├── run_eda.py                          # Main entry point to run EDA pipeline
├── README.md                           # Project documentation
└── requirements.txt                    # Pinned dependencies
```

---

## Validation Methodology & Fixed Random Seed
- **Validation Split**: Constructed strictly at the **Source 1 entity level** to prevent data leakage. All candidate pairs/records for a given Source 1 entity remain in the same split.
- **Split Ratio**: 80% Training (1,765,456 S1 entities) / 20% Validation (441,365 S1 entities).
- **Stratification**: Jointly stratified by ground-truth match count bucket (`0`, `1`, `2+` matches) and country (`US`, `India`).
- **Fixed Random Seed**: `42`.

---

## How to Run EDA & Profiling
To run the complete data loading, profiling, split generation, and infrastructure validation:
```bash
PYTHONPATH=. python3 run_eda.py
```

---

## How to Run Unit Tests
To execute the unit test suite covering data loading, schema validation, $F_{0.5}$ metric calculation, singletons, partial matches, country-level metrics, and blocking recall:
```bash
PYTHONPATH=. pytest tests/
```

---

## How to Run Evaluation Utilities

### 1. F0.5 Competition Metric Evaluator (`src/evaluation/metrics.py`)
```python
from src.evaluation.metrics import evaluate_predictions

# ground_truth: DataFrame with source1_entity_id & matched_entity_ids
# predictions: DataFrame with source1_entity_id & matched_entity_ids
# s1_data: DataFrame with entity_id & country

eval_results = evaluate_predictions(ground_truth_df, predictions_df, s1_df)

print("Overall F0.5:", eval_results['overall']['macro_f0_5'])
print("Country Breakdown:", eval_results['by_country'])
```

### 2. Blocking Recall Evaluator (`src/evaluation/measure_blocking_recall.py`)
```python
from src.evaluation.measure_blocking_recall import evaluate_blocking_recall

# candidate_generator: Callable(s1_df, s2_df, s3_df) -> Dict[str, Set[str]]
blocking_results = evaluate_blocking_recall(
    s1_df, s2_df, s3_df, ground_truth_df, candidate_generator, channel_name="my_channel"
)

print("Macro Blocking Recall:", blocking_results['macro_blocking_recall'])
print("Micro Blocking Recall:", blocking_results['micro_blocking_recall'])
print("Candidate Pair Count:", blocking_results['total_candidate_pairs'])
print("Reduction Ratio:", blocking_results['reduction_ratio'])
```

---

## External Data Statement
This project strictly adheres to competition integrity rules:
- **No external business databases** or government business registers are used.
- **No commercial or external Entity Resolution APIs** are used.
- **No geocoding or address normalization APIs** are used.
- **No internet data augmentation** is used.
- All normalization, blocking, and matching are trained and evaluated purely on the provided dataset files under `dataset/`.
