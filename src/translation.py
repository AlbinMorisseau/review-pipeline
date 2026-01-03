import polars as pl
import langid
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging

# Suppress noisy logs
logging.getLogger("langid").setLevel(logging.WARNING)

def detect_and_translate(df: pl.DataFrame, col_name: str, num_threads: int = 4) -> tuple[pl.DataFrame, int]:
    """
    Detects language and translates non-English rows to English.
    Returns the DataFrame and count of translated rows.
    """
    
    # 1. Detection
    def _detect(text: str) -> str:
        if not text or not text.strip(): return "unknown"
        return langid.classify(text)[0]

    texts = df[col_name].to_list()
    
    # Note: langid is fast enough that threading overhead might outweigh benefits on small data,
    # but kept parallel as per requirements.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        langs = list(tqdm(executor.map(_detect, texts), total=len(texts), desc="Detecting Language"))
        
    df = df.with_columns(pl.Series("detected_lang", langs))
    non_en_indices = [i for i, l in enumerate(langs) if l != 'en']
    
    if not non_en_indices:
        return df.drop("detected_lang"), 0

    # 2. Translation
    def _translate(text: str) -> str:
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except:
            return text # Fallback

    texts_to_trans = [texts[i] for i in non_en_indices]
    translated_subset = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        translated_subset = list(tqdm(
            executor.map(_translate, texts_to_trans), 
            total=len(texts_to_trans), 
            desc="Translating"
        ))
        
    # Merge back
    final_texts = texts.copy()
    for idx, trans_text in zip(non_en_indices, translated_subset):
        final_texts[idx] = trans_text
        
    return df.with_columns(pl.Series(col_name, final_texts)).drop("detected_lang"), len(non_en_indices)