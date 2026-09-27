import time
from typing import Dict, Set, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from src.normalization.text_normalizer import normalize_name, normalize_name_core

class TFIDFBlockingChannel:
    """
    Channel 2 Candidate Generator: Character n-gram TF-IDF sparse similarity retrieval.
    Fast precomputed name cache support, configurable n-grams, sublinear TF scaling, and batched target search.
    """
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        analyzer: str = 'char_wb',
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True,
        top_k: int = 10,
        batch_size: int = 5000,
        target_chunk_size: int = 1000000
    ):
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.top_k = top_k
        self.batch_size = batch_size
        self.target_chunk_size = target_chunk_size
        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            dtype=np.float32
        )
        
    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        name_cache: Optional[Dict[str, str]] = None
    ) -> Dict[str, Set[str]]:
        res = self.generate_candidates_multi_k(
            s1_df, s2_df, s3_df, k_list=[self.top_k], name_cache=name_cache
        )
        return res[self.top_k]

    def generate_candidates_multi_k(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        k_list: List[int] = [5, 10, 20, 50],
        name_cache: Optional[Dict[str, str]] = None
    ) -> Dict[int, Dict[str, Set[str]]]:
        t0 = time.time()
        max_k = max(k_list)
        
        if name_cache is None:
            names1 = s1_df['business_name'].dropna().values
            names2 = s2_df['business_name'].dropna().values if s2_df is not None else np.array([])
            names3 = s3_df['business_name'].dropna().values if s3_df is not None else np.array([])
            
            unique_names = np.unique(np.concatenate([names1, names2, names3]))
            name_cache = {n: normalize_name(n) for n in unique_names}
            name_cache[''] = ''
            
        s1_ids = s1_df['entity_id'].values
        s1_names = [name_cache.get(n, '') for n in s1_df['business_name'].fillna('').values]
        
        s2_ids = s2_df['entity_id'].values if s2_df is not None and len(s2_df) > 0 else np.array([])
        s2_names = [name_cache.get(n, '') for n in s2_df['business_name'].fillna('').values] if len(s2_ids) > 0 else []
        
        s3_ids = s3_df['entity_id'].values if s3_df is not None and len(s3_df) > 0 else np.array([])
        s3_names = [name_cache.get(n, '') for n in s3_df['business_name'].fillna('').values] if len(s3_ids) > 0 else []
        
        all_unique_normalized = np.unique(s1_names + s2_names + s3_names)
        if len(all_unique_normalized) == 0:
            return {k: {sid: set() for sid in s1_ids} for k in k_list}
            
        self.vectorizer.fit(all_unique_normalized)
        
        X_s1 = self.vectorizer.transform(s1_names)
        
        # Memory-bounded candidate accumulator (max_k elements per S1 ID)
        top_cands_with_scores: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in s1_ids}
        
        for target_ids, target_names in [(s2_ids, s2_names), (s3_ids, s3_names)]:
            if len(target_ids) == 0:
                continue
                
            n_target = len(target_ids)
            
            for target_start in range(0, n_target, self.target_chunk_size):
                target_end = min(target_start + self.target_chunk_size, n_target)
                chunk_ids = target_ids[target_start:target_end]
                chunk_names = target_names[target_start:target_end]
                
                X_target_chunk = self.vectorizer.transform(chunk_names)
                X_target_T = X_target_chunk.T.tocsc()
                
                n_s1 = len(s1_ids)
                for start_idx in range(0, n_s1, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_s1)
                    X_batch = X_s1[start_idx:end_idx]
                    
                    sim_batch = X_batch.dot(X_target_T).tocsr()
                    
                    for local_i in range(end_idx - start_idx):
                        global_i = start_idx + local_i
                        s1_id = s1_ids[global_i]
                        
                        row_start = sim_batch.indptr[local_i]
                        row_end = sim_batch.indptr[local_i + 1]
                        
                        if row_start == row_end:
                            continue
                            
                        indices = sim_batch.indices[row_start:row_end]
                        data = sim_batch.data[row_start:row_end]
                        
                        if len(data) > max_k:
                            top_local = np.argpartition(data, -max_k)[-max_k:]
                            chosen_indices = indices[top_local]
                            chosen_scores = data[top_local]
                        else:
                            chosen_indices = indices
                            chosen_scores = data
                            
                        existing = top_cands_with_scores[s1_id]
                        existing.extend(zip(chunk_ids[chosen_indices], [float(s) for s in chosen_scores]))
                        if len(existing) > max_k:
                            existing.sort(key=lambda x: x[1], reverse=True)
                            top_cands_with_scores[s1_id] = existing[:max_k]
                            
        # Slice top-K for each k in k_list
        res_multi: Dict[int, Dict[str, Set[str]]] = {k: {sid: set() for sid in s1_ids} for k in k_list}
        for s1_id, scored_list in top_cands_with_scores.items():
            if not scored_list:
                continue
            scored_list.sort(key=lambda x: x[1], reverse=True)
            for k in k_list:
                sliced = [cid for cid, _ in scored_list[:k]]
                res_multi[k][s1_id] = set(sliced)
                
        return res_multi

def generate_tfidf_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = 10,
    ngram_range: Tuple[int, int] = (3, 5),
    batch_size: int = 5000,
    name_cache: Optional[Dict[str, str]] = None
) -> Dict[str, Set[str]]:
    channel = TFIDFBlockingChannel(
        ngram_range=ngram_range,
        top_k=top_k,
        batch_size=batch_size,
        min_df=1
    )
    return channel.generate_candidates(s1_df, s2_df, s3_df, name_cache=name_cache)
