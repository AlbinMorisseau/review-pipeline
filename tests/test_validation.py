import pytest
from src.validation import compare_results

def test_compare_results_agreement():
    data = [{
        "chunk": "test",
        "kw_category": {"pet": "yes", "child": "no"},
        "bert_prediction": {"pet": "yes", "child": "no"}
    }]
    res = compare_results(data)
    assert res[0]["status"] == "kw_bert_agree"

def test_compare_results_disagreement():
    data = [{
        "chunk": "test",
        "kw_category": {"pet": "yes"},
        "bert_prediction": {"pet": "no"}
    }]
    res = compare_results(data)
    assert res[0]["status"] == "kw_bert_disputed"