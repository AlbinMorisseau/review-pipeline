import argparse
import os
import json
import torch
import polars as pl
from dotenv import load_dotenv
from pathlib import Path

from src.utils import setup_logger, load_json_config
from src.cleaning import basic_cleaning, convert_numbers, remove_special_characters, apply_stopwords_removal
from src.translation import detect_and_translate
from src.extraction import extract_categories, prepare_keywords
from src.chunking import merge_reviews_and_keywords, create_chunks
from src.inference import load_bert_model, run_inference
from src.validation import compare_results, get_stats
from src.export import export_category_csvs,export_complementary_csvs

load_dotenv()
NUM_THREADS = int(os.environ.get("NUM_THREADS", 6))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="End-to-End Review Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to original input CSV")
    parser.add_argument("--output", "-o", default="results/results.json", help="Path to final JSON output")
    parser.add_argument("--column", "-c",default="review", help="Review column name")
    parser.add_argument("--id_col", default="id", help="ID column name")
    parser.add_argument("--categories", default="data/categories.json")
    parser.add_argument("--exclusions", default="data/exclusions.json")
    parser.add_argument("--model_path", default="models/bert_finetuned", help="Path to fine-tuned BERT folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    logger = setup_logger()
    logger.info(f"Starting Pipeline on {DEVICE} with {NUM_THREADS} threads.")

    # STEP 1-3 - Cleaning and Keywords extraction
    dataset_name = Path(args.input).stem
    output_dir = Path(args.output).parent / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    kw_output_temp = output_dir / "temp_keywords_output.csv"
    json_output_path = output_dir / Path(args.output).name
    
    try:
        df = pl.read_csv(args.input)
        df, _, _ = basic_cleaning(df, args.column)
        df = convert_numbers(df, args.column)
        df = remove_special_characters(df, args.column)
        df, _ = detect_and_translate(df, args.column, NUM_THREADS)
        
        # Stopwords removal specific for keyword matching
        df_clean = apply_stopwords_removal(df, args.column)
        
        cats = load_json_config(args.categories)
        excls = load_json_config(args.exclusions)
        # Prepare lists
        cats_clean = prepare_keywords(cats)
        excls_clean = prepare_keywords(excls)
        
        # Extract keywords
        df_keywords = extract_categories(df_clean, args.column, cats_clean, excls_clean, args.id_col, NUM_THREADS)
        logger.info("Cleaning & Keywords extraction Done.")
        df_keywords.write_csv("data/results.csv")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    # STEP 4 - Chunking
    logger.info("Starting Chunking...")
    try:
        
        # Load Tokenizer for chunking (from model path)
        logger.info(f"Loading tokenizer from {args.model_path}")
        tokenizer, model = load_bert_model(args.model_path, DEVICE)
        
        # Raw Categories for regex matching inside chunks
        chunk_data = create_chunks(df_keywords, tokenizer, 128, cats, excls)
        logger.info(f"Created {len(chunk_data)} chunks.")
        
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        return

    # Step 5 - Validation with BERT
    logger.info("Starting Inference...")
    try:
        classes_list = list(cats.keys())

        results = run_inference(
            chunk_data, model, tokenizer, 
            classes=classes_list, 
            batch_size=args.batch_size, 
            threshold=args.threshold,
            device=DEVICE
        )
    except Exception as e:
        logger.error(f"Inference failed ({type(e).__name__}): {e}")
        raise

    # Step 6 - Statistics
    logger.info("Validating using BERT...")
    final_data = compare_results(results)
    stats = get_stats(final_data)
    logger.info(f"Validation Stats: {stats}")

    # Save Final JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    # STEP 7 - Export CSVs per category
    logger.info("Exporting CSV files...")
    
    export_category_csvs(
        data=final_data,
        output_dir=output_dir,
        categories_path= Path(args.categories)
    )

    export_complementary_csvs(
        df_reference=df_clean,
        id_col = args.id_col,
        review_col = args.column,
        category_csv_dir = output_dir,
        output_dir= output_dir
    )
    
    logger.info(f"Pipeline finished. Data saved to {args.output}")

if __name__ == "__main__":
    main()