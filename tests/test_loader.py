import os
import pytest
import pandas as pd
from src.data.loader import load_tsv, validate_schema, get_dataframe_stats

def test_load_tsv(tmp_path):
    tsv_file = tmp_path / "test.tsv"
    tsv_file.write_text("entity_id\tbusiness_name\tcountry\nS1-001\tAcme Corp\tUS\n")
    
    df = load_tsv(str(tsv_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == ["entity_id", "business_name", "country"]
    assert df.iloc[0]["entity_id"] == "S1-001"

def test_validate_schema():
    df = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Acme"],
        "business_address": ["Main St"],
        "country": ["US"]
    })
    expected = ["entity_id", "business_name", "business_address", "country"]
    assert validate_schema(df, expected, "test_dataset") is True
    
    with pytest.raises(ValueError):
        validate_schema(df, ["wrong", "cols"], "test_dataset")

def test_get_dataframe_stats():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": ["Acme", "", None],
        "business_address": ["123 Main St", "456 Oak Rd", "789 Pine Ave"],
        "country": ["US", "US", "India"]
    })
    stats = get_dataframe_stats(df, "sample_df")
    assert stats["row_count"] == 3
    assert stats["unique_entity_id_count"] == 3
    assert stats["duplicate_entity_id_count"] == 0
    assert stats["missing_values"]["business_name"] == 1
    assert stats["country_distribution"]["US"]["count"] == 2
    assert stats["country_distribution"]["India"]["count"] == 1
