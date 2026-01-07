import polars as pl
import re
from typing import List, Dict, Set
from transformers import PreTrainedTokenizerFast
from src.utils import make_regex

def merge_reviews_and_keywords(processed_csv: str, original_csv: str, id_col: str) -> pl.DataFrame:
    """Merges the keyword extraction result with the original review text."""
    kw_df = pl.read_csv(processed_csv)
    orig_df = pl.read_csv(original_csv)
    
    # Select only necessary columns from original
    orig_subset = orig_df.select([id_col, "review"])
    
    # Join
    df = kw_df.join(orig_subset, on=id_col, how="left")
    
    # Aggregate categories per ID
    df = df.group_by(id_col).agg([
        pl.col("review").first(),
        pl.col("category").cast(pl.Utf8).str.join(" ")
    ])
    return df

def _token_overlap_ratio(tokens_a: List[int], tokens_b: List[int]) -> float:
    set_a, set_b = set(tokens_a), set(tokens_b)
    inter = len(set_a & set_b)
    denom = min(len(set_a), len(set_b))
    return inter / denom if denom > 0 else 0.0

def _reduce_chunks(chunks: List[Dict], overlap_threshold: float = 0.75) -> List[Dict]:
    """Remove redundant chunks that overlap significantly."""
    if not chunks: return []
    
    used = set()
    final_chunks = []

    for i in range(len(chunks)):
        if i in used: continue
        
        # Form a group of overlapping chunks
        group = [chunks[i]]
        used.add(i)
        
        for j in range(i + 1, len(chunks)):
            if j in used: continue
            if _token_overlap_ratio(chunks[i]['tokens'], chunks[j]['tokens']) >= overlap_threshold:
                group.append(chunks[j])
                used.add(j)
        
        # Select best chunk in group (one with most keywords found inside)
        best = max(group, key=lambda c: c['kw_count'])
        final_chunks.append(best)
        
    return final_chunks

def create_chunks(
    df: pl.DataFrame, 
    tokenizer: PreTrainedTokenizerFast, 
    max_len: int, 
    keywords: Dict[str, List[str]], 
    exclusions: Dict[str, List[str]]
) -> List[Dict]:
    """
    Main logic: splits reviews into chunks around detected keywords.
    Returns a list of dictionaries ready for JSON export/Inference.
    """
    all_chunks = []
    all_cats = list(keywords.keys())

    # Pre-compile regex for performance
    kw_regex_map = {cat: [make_regex(k) for k in kws] for cat, kws in keywords.items()}
    ex_regex_map = {cat: [make_regex(e) for e in excls] for cat, excls in exclusions.items()}

    for row in df.iter_rows(named=True):
        text = row["review"]
        row_cats = set(row["category"].split()) if row["category"] else set()
        
        if not text or not row_cats: continue

        # Tokenize full text to find offsets
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
        input_ids = encoding.input_ids
        offsets = encoding.offset_mapping
        
        trigger_indices = set()

        # 1. Find trigger tokens (where keywords are)
        for cat in row_cats:
            if cat not in keywords: continue
            
            # Exclusion spans
            ex_spans = []
            for pat in ex_regex_map.get(cat, []):
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    ex_spans.append(m.span())

            # Keyword matches
            for pat in kw_regex_map.get(cat, []):
                for match in re.finditer(pat, text, flags=re.IGNORECASE):
                    start, end = match.span()
                    # Check exclusion
                    if any(est <= start and een >= end for est, een in ex_spans):
                        continue
                    
                    # Map char to token
                    t_idx = encoding.char_to_token(start)
                    # Fallback if char_to_token returns None (rare edge cases)
                    if t_idx is None:
                        for i, (os, oe) in enumerate(offsets):
                            if os <= start < oe:
                                t_idx = i
                                break
                    
                    if t_idx is not None:
                        trigger_indices.add(t_idx)

        # 2. Create raw chunks around triggers
        raw_chunks = []
        sorted_indices = sorted(trigger_indices)
        
        for center_idx in sorted_indices:
            half = max_len // 2
            start = max(0, center_idx - half)
            end = min(len(input_ids), center_idx + half)
            
            # Adjust window if too short
            if end - start < max_len:
                if start == 0: end = min(len(input_ids), max_len)
                else: start = max(0, len(input_ids) - max_len)
            
            chunk_ids = input_ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            
            # Re-verify keywords in chunk
            kw_presence = {c: "no" for c in all_cats}
            kw_count = 0
            for cat, patterns in kw_regex_map.items():
                for pat in patterns:
                    if re.search(pat, chunk_text, flags=re.IGNORECASE):
                         kw_presence[cat] = "yes"
                         kw_count += 1
                         break 
            
            raw_chunks.append({
                "text": chunk_text,
                "tokens": chunk_ids,
                "kw_category": kw_presence,
                "kw_count": kw_count
            })

        # 3. Deduplicate
        unique_chunks = _reduce_chunks(raw_chunks)
        
        for ch in unique_chunks:
            all_chunks.append({
                "original_id": row["id"],
                "review" : row["review"],
                "review_cleaned" : row["review_cleaned"],
                "chunk": ch["text"],
                "keywords_found": row["keywords_found"],
                "kw_category": ch["kw_category"],
            })
            
    return all_chunks