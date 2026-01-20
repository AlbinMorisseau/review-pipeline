import re
import ollama
import json
import argparse
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import setup_logger

random.seed(42)
np.random.seed(42)
# Configure logging
logger = setup_logger(name="needs_extractor")

class NeedExtractor:
    """
    Extracts, clusters, refines, and assigns user needs (topics) from review data.
    """

    def __init__(self, gen_model: str = "mistral", embed_model: str = "nomic-embed-text"):
        self.gen_model = gen_model
        self.embed_model = embed_model

    def _embed(self, text: str) -> np.ndarray:
        """Generates embedding for a given text using Ollama."""
        if not text or len(text) < 3:
            return np.zeros(768)
        try:
            res = ollama.embeddings(model=self.embed_model, prompt=text)
            return np.array(res["embedding"])
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.zeros(768)
    
    def _safe_json_load(self,text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")
        return json.loads(match.group())

    def _llm_json(self, prompt: str) -> Dict[str, Any]:
        """Queries the LLM and expects a JSON response."""
        try:
            res = ollama.chat(
                model=self.gen_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_predict": 2000}
            )
            content = res["message"]["content"]
            return self._safe_json_load(content)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON. Returning empty dict.")
            return {}
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {}


    def extract_topics_from_reviews(
        self,
        df: pd.DataFrame,
        text_col: str,
        keyword_col: str,
        context_category: str,
        batch_size: int = 15
    ) -> List[str]:
        """
        Extracts raw topics from reviews batch by batch using the LLM.
        """
        logger.info("Starting topic extraction from reviews...")
        all_topics = []
        work_df = df.copy()
        
        # Ensure keyword column handles NaNs if present
        if keyword_col in work_df.columns:
            work_df[keyword_col] = work_df[keyword_col].fillna("misc")
        else:
            work_df[keyword_col] = "misc"

        batches = []
        # Group by keywords to give context-aware batches, or just chunk if no keywords
        for _, group in work_df.groupby(keyword_col):
            reviews = group[text_col].tolist()
            for i in range(0, len(reviews), batch_size):
                batches.append(reviews[i:i + batch_size])

        for batch in tqdm(batches, desc="Extracting raw topics"):
            prompt = f"""
                You analyze reviews from travelers with specific needs ({context_category}).

                Extract 2–7 GENERIC and SHAREABLE user needs.
                Ignore anecdotal or one-off details.
                Avoid extremely generic topics.
                Focus on specific user needs.

                Rules:
                - 4–8 words per topic
                - Small justification sentence
                - Focus on needs, not sentiment

                Reviews:
                {json.dumps(batch)}

                Output:
                {{ "topics": ["topic : justification"] }}
                """
            res = self._llm_json(prompt)
            all_topics.extend(res.get("topics", []))

        # Clean strings
        clean_topics = []
        for t in all_topics:
            if isinstance(t, str):
                clean_topics.append(t)
            elif isinstance(t, dict):
                # Handle edge case where LLM returns list of dicts instead of strings
                clean_topics.append(f"{t.get('topic', '')} : {t.get('justification', '')}")

        unique_topics = list(dict.fromkeys(clean_topics))
        logger.info(f"Extracted {len(unique_topics)} unique raw topics.")
        return unique_topics

    def cluster_and_reduce_topics(
        self,
        topics: List[str],
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.85
    ) -> List[str]:
        """
        Reduces the list of topics using embedding-based agglomerative clustering.
        """
        logger.info("Clustering topics to reduce redundancy...")
        if not topics:
            return []

        embeddings = np.array([self._embed(t) for t in topics])

        norms = np.linalg.norm(embeddings, axis=1)
        valid_mask = norms > 0

        filtered_topics = [t for t, v in zip(topics, valid_mask) if v]
        filtered_embeddings = embeddings[valid_mask]

        logger.info(
            f"Removed {len(topics) - len(filtered_topics)} topics with zero embeddings"
        )

        if len(filtered_topics) < min_cluster_size:
            logger.warning("Not enough valid topics after filtering")
            return []

        # Distance threshold is 1 - similarity (cosine distance)
        distance_threshold = 1 - similarity_threshold
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="complete",
            distance_threshold=distance_threshold
        )

        labels = clustering.fit_predict(filtered_embeddings)

        topic_to_embedding = {
            t: e for t, e in zip(filtered_topics, filtered_embeddings)
        }


        clusters = {}
        for topic, label in zip(filtered_topics, labels):

            clusters.setdefault(label, []).append(topic)

        reduced_topics = []
        for group in clusters.values():
            # Only keep clusters that have enough 'weight' (frequency)
            if len(group) < min_cluster_size:
                continue

            # Find the centroid (most representative topic) of the cluster
            vecs = np.array([topic_to_embedding[t] for t in group])
            sim_matrix = cosine_similarity(vecs)

            # Index of the topic with highest average similarity to others in group
            central_idx = int(sim_matrix.mean(axis=1).argmax())
            reduced_topics.append(group[central_idx])

        logger.info(f"Reduced topics from {len(topics)} to {len(reduced_topics)} via clustering.")
        return list(set(reduced_topics))

    def assign_topics_to_reviews_hybrid(
        self,
        reviews: List[str],
        topics: List[str],
        embedding_threshold: float = 0.45,
        tfidf_threshold: float = 0.08
    ) -> pd.DataFrame:
        """
        Assigns the best matching topic to each review using a hybrid approach (Embedding + TF-IDF).
        """
        logger.info("Assigning topics to reviews...")
        
        if not topics:
            logger.error("No topics available for assignment.")
            return pd.DataFrame()

        # --- Embeddings Calculation ---
        topic_embeds = np.array([self._embed(t) for t in topics])
        review_embeds = np.array([self._embed(r) for r in reviews])
        
        # Matrix: [n_reviews, n_topics]
        embed_sim = cosine_similarity(review_embeds, topic_embeds)

        # --- TF-IDF Calculation ---
        corpus = reviews + topics
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=10000
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        review_tfidf = tfidf_matrix[:len(reviews)]
        topic_tfidf = tfidf_matrix[len(reviews):]
        
        tfidf_sim = cosine_similarity(review_tfidf, topic_tfidf)

        rows = []
        for i, review in enumerate(tqdm(reviews, desc="Assigning")):
            
            # Get best embedding match
            emb_idx = embed_sim[i].argmax()
            emb_score = embed_sim[i][emb_idx]

            # Get best TF-IDF match
            tfidf_idx = tfidf_sim[i].argmax()
            tfidf_score = tfidf_sim[i][tfidf_idx]

            # Decision Logic
            if emb_score >= embedding_threshold:
                assigned_topic = topics[emb_idx]
                method = "embedding"
                final_score = emb_score
            elif tfidf_score >= tfidf_threshold:
                assigned_topic = topics[tfidf_idx]
                method = "tfidf"
                final_score = tfidf_score
            
            else:
                assigned_topic = "No dominant need"
                method = "none"
                final_score = max(emb_score, tfidf_score)

            rows.append({
                "review": review,
                "assigned_topic": assigned_topic,
                "score": round(float(final_score), 3),
                "method": method
            })

        return pd.DataFrame(rows)

    def generate_representative_json(self, result_df: pd.DataFrame, min_need_number: int = 2) -> List[Dict]:
        """Generates the final representative JSON output."""
        logger.info("Generating final JSON summary...")
        
        total_reviews = len(result_df)

        topic_counts = result_df["assigned_topic"].value_counts()
        kept_topics = [t for t, c in topic_counts.items() if c >= min_need_number and t != "No dominant need"]

        json_output = []

        for topic in kept_topics:
            topic_reviews = result_df[result_df["assigned_topic"] == topic]["review"].tolist()

            # Rank reviews by similarity to the topic string itself for "representativeness"
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            try:
                tfidf = vectorizer.fit_transform(topic_reviews + [topic])
                scores = cosine_similarity(tfidf[:-1], tfidf[-1]).flatten()
                
                # Pair review with score
                ranked = sorted(
                    zip(topic_reviews, scores),
                    key=lambda x: x[1],
                    reverse=True
                )[:3] # Keep top 3

                json_output.append({
                    "topic": topic,
                    "count": int(topic_counts[topic]),
                    "representative_reviews": [
                        {"review": r, "tfidf_score": round(float(s), 3)} for r, s in ranked
                    ]
                })
            except ValueError:
                # Handle cases with not enough data for TF-IDF
                continue

        return json_output

def main():
    parser = argparse.ArgumentParser(description="Extract and Assign User Needs from Reviews using LLM and embedding")
    
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--text_col", default="chunk", help="Column name containing chunked reviews. It might work with full reviews but results might be noisy.")
    parser.add_argument("--keyword_col", default="keywords_found", help="Column for grouping (optional)")
    parser.add_argument("--category", required=True, help="Context category (e.g., 'pets', 'handicap', 'children')")
    parser.add_argument("--batch_size", default=15, help="Number of reviews given per LLM call for raw needs extraction")
    parser.add_argument("--sim_threshold", default=0.5, help="Treshold used to merge similar topics. Clustering done with distance = 1 - sim_treshold")
    parser.add_argument("--min_cluster_size", default=2, help="Number of topics by cluster to be considered relevant enough")
    parser.add_argument("--embed_threshold", default=0.5, help="Threshold to define if an embedded review can be associated to a topic/need")
    parser.add_argument("--tfidf_threshold", default=0.1, help="Threshold to define if a review can be associated to a topic/need using TFIDF")
    parser.add_argument("--min_needs_number", default=6, help= "Control the number of reviews that has to be linked to a topic to considere this topic relevant")
    parser.add_argument("--gen_model", default="mistral", help="Ollama generation model")
    parser.add_argument("--embed_model", default="nomic-embed-text", help="Ollama embedding model")

    args = parser.parse_args()

    # Creation of output directory
    OUTPUT_DIR = "results/topic_modelling/needs_extraction"
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"reviews_needs_{args.category}.csv"
    output_json = output_dir / f"needs_summary_{args.category}.json"
   
    # Load Data
    try:
        df = pd.read_csv(args.input)
        df = df.dropna(subset=[args.text_col])
        logger.info(f"Loaded {len(df)} rows from {args.input}")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return

    engine = NeedExtractor(gen_model=args.gen_model, embed_model=args.embed_model)

    # 1. Extraction
    raw_topics = engine.extract_topics_from_reviews(
        df=df,
        batch_size=args.batch_size,
        text_col=args.text_col,
        keyword_col=args.keyword_col,
        context_category=args.category
    )

    # 2. Clustering
    clustered_topics = engine.cluster_and_reduce_topics(
        raw_topics,
        min_cluster_size=2,
        similarity_threshold=0.50
    )

    # 3. Assignation
    result_df = engine.assign_topics_to_reviews_hybrid(
        reviews=df[args.text_col].tolist(),
        topics=clustered_topics,
        embedding_threshold=0.40,
        tfidf_threshold=0.08
    )

    # Save Results
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Detailed CSV saved to {output_dir}")

    summary_json = engine.generate_representative_json(result_df=result_df,min_need_number=8)
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    logger.info(f"Summary JSON saved to {output_json}")

if __name__ == "__main__":
    main()