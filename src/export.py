import polars as pl
from pathlib import Path
from typing import List, Dict
import logging
import json

logger = logging.getLogger(__name__)

def export_category_csvs(data: List[Dict], output_dir: Path, categories_path: Path):
    """
    Generate a CSV file containing each review related to a category
    Columns : original_id, review, review_cleaned, chunk, keywords_found, status.
    """

    generated_files = 0
    total_rejected = 0

    with categories_path.open("r", encoding="utf-8") as f:
        categories_config = json.load(f)

    categories = [
        key
        for key, value in categories_config.items()
        if not (isinstance(value, list) and len(value) == 0)
    ]

    for category in categories:

        allowed_keywords = set(categories_config[category])

        filtered_rows = []
        rejected_for_category = 0

        for row in data:
            if row.get("bert_prediction", {}).get(category) != "yes":
                continue

            keywords_str = row.get("keywords_found", "")
            found_keywords = {
                kw.strip() for kw in keywords_str.split(",") if kw.strip()
            }

            # Filtering : at least a keyword related to the given category
            if found_keywords & allowed_keywords:
                filtered_rows.append({
                    "original_id": row["original_id"],
                    "review": row["review"],
                    "review_cleaned": row["review_cleaned"],
                    "chunk": row["chunk"],
                    "keywords_found": row["keywords_found"],
                    "status": row.get("status", "unknown")
                })
            else:
                rejected_for_category += 1

        if not filtered_rows:
            continue

        df = pl.DataFrame(
            filtered_rows,
            schema={
                "original_id": pl.Int64,
                "review": pl.Utf8,
                "review_cleaned": pl.Utf8,
                "chunk": pl.Utf8,
                "keywords_found": pl.Utf8,
                "status": pl.Utf8,
            }
        )

        file_path = output_dir / f"{category}.csv"
        df.write_csv(file_path)

        generated_files += 1

    logger.info(
        f"CSV creation ended : {generated_files} files created in '{output_dir}'. "
    )
