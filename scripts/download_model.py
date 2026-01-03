from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

MODEL_NAME = "AlbinMorisseau/bert-finetuned-review"
TARGET_DIR = Path("../models/bert_finetuned")

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(TARGET_DIR)

    print("Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(TARGET_DIR)

    print(f"Model downloaded to {TARGET_DIR.resolve()}")

if __name__ == "__main__":
    main()
    