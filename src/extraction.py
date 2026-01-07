import polars as pl
import re
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.cleaning import remove_stopwords_from_text

def prepare_keywords(data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Clean stopwords from category/exclusion keywords list."""
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, list):
            cleaned[k] = [remove_stopwords_from_text(i) for i in v]
        else:
            cleaned[k] = v
    return cleaned

def extract_categories(
    df: pl.DataFrame, 
    col_name: str, 
    categories: Dict[str, List[str]], 
    exclusions: Dict[str, List[str]], 
    id_col: str,
    num_threads: int = 4
) -> pl.DataFrame:
    """
    Matches keywords against text, considering exclusions.
    Returns filtered DataFrame.
    """
    
    # Helper for regex compilation
    def make_regex(kw: str) -> str:
        kw = kw.strip().replace(" - ", "-")
        if " " in kw or "-" in kw:
            # Handle phrases
            return re.escape(kw).replace(r"\-", r"[-\s]").replace(r"\ ", r"\s+")
        return r"\b" + re.escape(kw) + r"\b"

    # Core logic per category
    def process_category(cat_name, keywords, excluded_phrases, data_rows):
        results = []
        
        # Pre-compile regexes for speed
        ex_patterns = [re.compile(make_regex(ex), re.IGNORECASE) for ex in excluded_phrases] if excluded_phrases else []
        kw_patterns = [(kw, re.compile(make_regex(kw), re.IGNORECASE)) for kw in keywords]

        for row in data_rows:
            text = row[col_name]
            text_cleaned = row[col_name + "_cleaned"]
            rid = row[id_col]
            
            if not isinstance(text, str): continue
            
            temp_text = text
            # Mask exclusions
            for pat in ex_patterns:
                temp_text = pat.sub(" ", temp_text)
            
            # Check match
            found = [kw for kw, pat in kw_patterns if pat.search(temp_text)]
            
            if found:
                results.append((rid, text, text_cleaned, ", ".join(found), cat_name))
        return results

    # Execution
    alias = col_name + "_cleaned"
    rows = df.select([id_col, col_name,alias]).to_dicts()
    all_results = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(process_category, cat, kws, exclusions.get(cat, []), rows): cat
            for cat, kws in categories.items()
        }
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Keyword Extraction"):
            all_results.extend(fut.result())

    if not all_results:
        return pl.DataFrame(schema={id_col: pl.Int64, "review": pl.Utf8, "review_cleaned": pl.Utf8, "keywords": pl.Utf8, "category": pl.Utf8})

    return pl.DataFrame(
        all_results, 
        schema=[id_col, "review", "review_cleaned", "keywords_found", "category"], 
        orient="row"
    )