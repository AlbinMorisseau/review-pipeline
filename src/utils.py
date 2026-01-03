import logging
import os
import json
from typing import Dict, List
import re

def setup_logger(name: str = "pipeline", log_dir: str = "logs") -> logging.Logger:
    """Configures a professional logger with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    # File Handler
    fh = logging.FileHandler(os.path.join(log_dir, "pipeline.log"))
    fh.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
        
    return logger

def load_json_config(path: str) -> Dict[str, List[str]]:
    """Loads a JSON configuration file safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def normalize_keyword(kw: str) -> str:
    return kw.strip().replace(" - ", "-")

def make_regex(kw: str) -> str:
    """Creates a regex pattern for a keyword, handling phrases and boundaries."""
    kw = normalize_keyword(kw)
    if " " in kw or "-" in kw:
        return re.escape(kw).replace(r"\-", r"[-\s]").replace(r"\ ", r"\s+")
    else:
        return r"\b" + re.escape(kw) + r"\b"

