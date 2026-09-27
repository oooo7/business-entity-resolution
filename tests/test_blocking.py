import pytest
import pandas as pd
from src.blocking.name_country_blocking import generate_name_country_candidates
from src.blocking.tfidf_blocking import generate_tfidf_candidates

def test_name_country_blocking_basic():
    s1_df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Acme Corp Pvt Ltd", "Beta Technologies"],
        "country": ["US", "India"]
    })
    s2_df = pd.DataFrame({
        "entity_id": ["S2-10", "S2-20"],
        "business_name": ["Acme Corp", "Beta Tech"],
        "country": ["US", "India"]
    })
    s3_df = pd.DataFrame({
        "entity_id": ["S3-100"],
        "business_name": ["Beta Technologies Inc"],
        "country": ["India"]
    })
    
    cands = generate_name_country_candidates(s1_df, s2_df, s3_df)
    
    # S1-1 core_name = 'acme', country = 'US' -> matches S2-10
    assert "S2-10" in cands["S1-1"]
    # S1-2 core_name = 'beta technologies', country = 'India' -> matches S3-100
    assert "S3-100" in cands["S1-2"]
    assert "S2-10" not in cands["S1-2"]

def test_tfidf_blocking_configurable_k():
    s1_df = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Starbucks Coffee Company"],
        "country": ["US"]
    })
    s2_df = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S2-3"],
        "business_name": ["Starbucks Coffee", "Starbucks Store", "Dunkin Donuts"],
        "country": ["US", "US", "US"]
    })
    s3_df = pd.DataFrame({
        "entity_id": ["S3-1"],
        "business_name": ["Starbucks Express"],
        "country": ["US"]
    })
    
    # K = 1
    cands_k1 = generate_tfidf_candidates(s1_df, s2_df, s3_df, top_k=1)
    # K = 2
    cands_k2 = generate_tfidf_candidates(s1_df, s2_df, s3_df, top_k=2)
    
    assert len(cands_k1["S1-1"]) <= 2  # At most 1 from S2 and 1 from S3
    assert "S2-1" in cands_k1["S1-1"]
    assert len(cands_k2["S1-1"]) >= len(cands_k1["S1-1"])

def test_tfidf_blocking_no_source1_or_duplicate_ids():
    s1_df = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Acme Corp"],
        "country": ["US"]
    })
    s2_df = pd.DataFrame({
        "entity_id": ["S2-1", "S2-1"], # Duplicate row in source input
        "business_name": ["Acme Corp", "Acme Corp"],
        "country": ["US", "US"]
    })
    s3_df = pd.DataFrame({"entity_id": [], "business_name": [], "country": []})
    
    cands = generate_tfidf_candidates(s1_df, s2_df, s3_df, top_k=5)
    
    # Returned set must be unique and contain no S1-1 IDs
    for cid in cands["S1-1"]:
        assert cid.startswith("S2-") or cid.startswith("S3-")
        assert not cid.startswith("S1-")
