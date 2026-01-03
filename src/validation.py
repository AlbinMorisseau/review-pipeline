from collections import Counter
from typing import List, Dict

def compare_results(data: List[Dict]) -> List[Dict]:
    """Adds a 'status' field: kw_bert_agree or kw_bert_disputed."""
    for item in data:
        kw = item.get("kw_category", {})
        bert = item.get("bert_prediction", {})

        if kw == bert:
            item["status"] = "kw_bert_agree"
        else:
            item["status"] = "kw_bert_disputed"
    return data

def get_stats(data: List[Dict]) -> Dict[str, int]:
    """Returns basic stats about agreement."""
    c = Counter(item.get("status", "unknown") for item in data)
    return dict(c)