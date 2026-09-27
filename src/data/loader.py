import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

EXPECTED_SOURCE_SCHEMA = ['entity_id', 'business_name', 'business_address', 'country']
EXPECTED_GT_SCHEMA = ['source1_entity_id', 'matched_entity_ids']

def load_tsv(filepath: str) -> pd.DataFrame:
    """
    Load a tab-separated (.tsv) file using pd.read_csv with explicit tab separator.
    Rule 8 requirement.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, sep="\t")

def validate_schema(df: pd.DataFrame, expected_cols: List[str], dataset_name: str) -> bool:
    """
    Validate that a dataframe contains the expected columns.
    """
    actual_cols = list(df.columns)
    if actual_cols != expected_cols:
        raise ValueError(
            f"Schema mismatch for {dataset_name}.\n"
            f"Expected: {expected_cols}\n"
            f"Actual:   {actual_cols}"
        )
    return True

def load_all_datasets(data_dir: str = "dataset") -> Dict[str, pd.DataFrame]:
    """
    Load all 7 dataset files (3 train sources + ground truth, 3 test sources)
    and validate their schemas.
    """
    files = {
        'train_source1': (os.path.join(data_dir, 'train/train_source1.tsv'), EXPECTED_SOURCE_SCHEMA),
        'train_source2': (os.path.join(data_dir, 'train/train_source2.tsv'), EXPECTED_SOURCE_SCHEMA),
        'train_source3': (os.path.join(data_dir, 'train/train_source3.tsv'), EXPECTED_SOURCE_SCHEMA),
        'train_ground_truth': (os.path.join(data_dir, 'train/train_ground_truth.tsv'), EXPECTED_GT_SCHEMA),
        'test_source1': (os.path.join(data_dir, 'test/test_source1.tsv'), EXPECTED_SOURCE_SCHEMA),
        'test_source2': (os.path.join(data_dir, 'test/test_source2.tsv'), EXPECTED_SOURCE_SCHEMA),
        'test_source3': (os.path.join(data_dir, 'test/test_source3.tsv'), EXPECTED_SOURCE_SCHEMA),
    }
    
    datasets = {}
    for name, (path, schema) in files.items():
        df = load_tsv(path)
        validate_schema(df, schema, name)
        datasets[name] = df
        
    return datasets

def get_dataframe_stats(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics for Part A requirements.
    """
    n_rows = len(df)
    id_col = 'entity_id' if 'entity_id' in df.columns else 'source1_entity_id'
    
    n_unique_ids = int(df[id_col].nunique())
    n_dup_ids = int(n_rows - n_unique_ids)
    
    missing_vals = {col: int(df[col].isna().sum()) for col in df.columns}
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    empty_strings = {}
    string_stats = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            s_str = df[col].dropna().astype(str)
            # Count empty strings or whitespace-only
            empty_cnt = int((s_str.str.strip() == '').sum() + df[col].isna().sum())
            empty_strings[col] = empty_cnt
            
            lens = s_str.str.len()
            if len(lens) > 0:
                string_stats[col] = {
                    'min_length': int(lens.min()),
                    'median_length': float(lens.median()),
                    'mean_length': float(lens.mean()),
                    'max_length': int(lens.max()),
                    'missing_or_empty_pct': float(empty_cnt / n_rows * 100)
                }
            else:
                string_stats[col] = {
                    'min_length': 0, 'median_length': 0.0, 'mean_length': 0.0, 'max_length': 0,
                    'missing_or_empty_pct': 100.0
                }
                
    country_dist = {}
    if 'country' in df.columns:
        vc = df['country'].value_counts(dropna=False).to_dict()
        for k, v in vc.items():
            key_str = str(k) if pd.notna(k) else "MISSING"
            country_dist[key_str] = {
                'count': int(v),
                'percentage': float(v / n_rows * 100)
            }
            
    return {
        'name': name,
        'row_count': n_rows,
        'column_names': list(df.columns),
        'dtypes': dtypes,
        'unique_entity_id_count': n_unique_ids,
        'duplicate_entity_id_count': n_dup_ids,
        'missing_values': missing_vals,
        'empty_strings': empty_strings,
        'country_distribution': country_dist,
        'string_stats': string_stats
    }
