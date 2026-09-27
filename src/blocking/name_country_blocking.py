import time
from typing import Dict, Set, List, Optional
import numpy as np
import pandas as pd
from src.normalization.text_normalizer import normalize_name_core

def generate_name_country_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    name_core_cache: Optional[Dict[str, str]] = None
) -> Dict[str, Set[str]]:
    """
    Channel 1 Candidate Generator: (normalized_name_core, country) inverted index lookup.
    """
    t0 = time.time()
    
    if name_core_cache is None:
        names1 = s1_df['business_name'].dropna().values
        names2 = s2_df['business_name'].dropna().values if s2_df is not None else np.array([])
        names3 = s3_df['business_name'].dropna().values if s3_df is not None else np.array([])
        
        unique_names = np.unique(np.concatenate([names1, names2, names3]))
        name_core_cache = {n: normalize_name_core(n) for n in unique_names}
        name_core_cache[''] = ''
        
    index: Dict[tuple, Set[str]] = {}
    
    for df in [s2_df, s3_df]:
        if df is None or len(df) == 0:
            continue
            
        e_ids = df['entity_id'].values
        b_names = df['business_name'].fillna('').values
        countries = df['country'].fillna('').values
        
        for eid, raw_name, country in zip(e_ids, b_names, countries):
            c_name = name_core_cache.get(raw_name, '')
            if not c_name:
                continue
            key = (c_name, str(country).strip())
            if key not in index:
                index[key] = set()
            index[key].add(eid)
            
    s1_ids = s1_df['entity_id'].values
    s1_names = s1_df['business_name'].fillna('').values
    s1_countries = s1_df['country'].fillna('').values
    
    candidates_dict: Dict[str, Set[str]] = {}
    for s1_id, raw_name, country in zip(s1_ids, s1_names, s1_countries):
        c_name = name_core_cache.get(raw_name, '')
        if not c_name:
            candidates_dict[s1_id] = set()
            continue
            
        key = (c_name, str(country).strip())
        cand_set = index.get(key, set())
        candidates_dict[s1_id] = set(cand_set)
        
    return candidates_dict
