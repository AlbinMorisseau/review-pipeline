import polars as pl
import re
from num2words import num2words
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk

# Ensure NLTK data is present
nltk.download("stopwords", quiet=True)

# Compile Regex patterns once
CONTRACTIONS = [
    "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't",
    "hasn't", "haven't", "he's", "I'm", "I've", "isn't", "it's", "she's",
    "shouldn't", "that's", "there's", "they're", "they've", "we're", "we've",
    "weren't", "what's", "where's", "who's", "won't", "wouldn't", "you're", "you've"
]
# Double pattern to catch straight and curly quotes
CONTRACTIONS_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in CONTRACTIONS) + 
    r'|' + '|'.join(re.escape(c.replace("'", "’")) for c in CONTRACTIONS) + r')\b', 
    re.IGNORECASE
)

STOP_WORDS = set(stopwords.words("english"))
TOKENIZER = TweetTokenizer(preserve_case=False)

def basic_cleaning(df: pl.DataFrame, col_name: str) -> tuple[pl.DataFrame, int, int]:
    """Removes nulls and duplicates."""
    initial_count = df.height
    df = df.drop_nulls(subset=[col_name])
    missing_count = initial_count - df.height
    
    current_count = df.height
    df = df.unique(subset=[col_name])
    duplicate_count = current_count - df.height
    
    return df, missing_count, duplicate_count

def convert_numbers(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """Converts numerical digits to words."""
    def _replace(text: str) -> str:
        if not text: return text
        return re.sub(r'\d+', lambda m: num2words(int(m.group())), text)

    return df.with_columns(
        pl.col(col_name).map_elements(_replace, return_dtype=pl.Utf8).alias(col_name)
    )

def remove_special_characters(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """Cleans text, removing URLs/Emails but preserving contractions."""
    
    def _clean(text: str) -> str:
        if not isinstance(text, str): return ""
        
        # 1. Protect Contractions
        contractions_map = {}
        def replace_contraction(m):
            key = f"__CT_{len(contractions_map)}__"
            contractions_map[key] = m.group(0)
            return key
        text = CONTRACTIONS_PATTERN.sub(replace_contraction, text)
        
        # 2. Regex Cleaning
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text) # Email
        text = re.sub(r"https?://\S+|www\.\S+", "", text) # URL
        text = re.sub(r"#\S+", "", text) # Hashtag
        text = re.sub(r"(\w)[’']s\b", r"\1 s", text) # Possessive
        text = text.replace("/", " ").replace("-", " ")
        text = re.sub(r"[^\w\s']", " ", text) # Keep words, spaces, quotes
        
        # 3. Restore Contractions
        for k, v in contractions_map.items():
            text = text.replace(k, v)
            
        return re.sub(r"\s+", " ", text).strip()

    return df.with_columns(
        pl.col(col_name).map_elements(_clean, return_dtype=pl.Utf8).alias(col_name)
    )

def remove_stopwords_from_text(text: str, keep_exceptions: bool = True) -> str:
    """Helper to remove stopwords from a single string."""
    if not text: return ""
    text = text.replace("’", "'")
    tokens = TOKENIZER.tokenize(text)
    
    # Simple valid token check inside list comp for speed
    filtered = [
        t for t in tokens 
        if not t.startswith("'") and re.match(r"^[a-z]+(?:[-'][a-z]+)*$", t)
        and (t not in STOP_WORDS or (keep_exceptions and t in CONTRACTIONS)) # Simplified exception logic
    ]
    return " ".join(filtered)

def apply_stopwords_removal(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """Applies stopwords removal on a DataFrame column and keeps the original."""
    alias = col_name + "_cleaned"
    return df.with_columns(
        pl.col(col_name)
        .map_elements(remove_stopwords_from_text, return_dtype=pl.Utf8)
        .alias(alias)
    )