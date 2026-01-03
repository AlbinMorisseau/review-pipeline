import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

def load_bert_model(model_path: str, device: str):
    """Loads tokenizer and model from local path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

def run_inference(
    data: List[Dict], 
    model, 
    tokenizer, 
    classes: List[str], 
    batch_size: int = 16, 
    threshold: float = 0.5,
    device: str = "cpu"
) -> List[Dict]:
    """
    Runs BERT inference on the 'chunk' field of the input data list.
    Updates the list in-place with 'bert_prediction'.
    """
    id2label = {i: label for i, label in enumerate(classes)}
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="BERT Inference"):
        batch = data[i : i + batch_size]
        texts = [item["chunk"] for item in batch]
        
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            
        for item, prob_vec in zip(batch, probs):
            pred = {
                id2label[idx]: ("yes" if p >= threshold else "no")
                for idx, p in enumerate(prob_vec)
            }
            item["bert_prediction"] = pred
            
    return data