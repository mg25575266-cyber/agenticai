import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Change working directory to the script's location
# Ensures that csv files etc below are in the same directory as code
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = "customer_service_training_data.csv"
EMBED_PATH = "intent_embeddings.npy"
META_PATH = "intent_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

df = pd.read_csv(DATA_PATH)
utterances = df["utterance"].astype(str).tolist()
print(f"Loaded {len(utterances)} utterances from {DATA_PATH}")

if os.path.exists(EMBED_PATH) and os.path.exists(META_PATH):
    print("Loading existing embeddings from disk...")
    utterance_embeddings = np.load(EMBED_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    print(f"Loaded {len(utterance_embeddings)} embeddings.")
else:
    print("Generating embeddings (first time)...")
    utterance_embeddings = model.encode(
        utterances,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Save both embeddings and metadata
    np.save(EMBED_PATH, utterance_embeddings)
    meta = {
        "utterances": utterances,
        "intents": df["intent"].tolist(),
        "categories": df["category"].tolist()
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved embeddings to {EMBED_PATH} and metadata to {META_PATH}")

# Test embeddings
def classify_intent_semantic(user_query: str):
    query_emb = model.encode([user_query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, utterance_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    return {
        "intent": meta["intents"][best_idx],
        "category": meta["categories"][best_idx],
        "match_utterance": meta["utterances"][best_idx],
        "similarity": round(float(similarities[best_idx]), 3)
    }

if __name__ == "__main__":
    test_queries = [
        "How do I open a new account?",
        "Can I use one email for multiple accounts?",
        "I want a refund for my order",
    ]

    for q in test_queries:
        result = classify_intent_semantic(q)
        print(f"\nUser: {q}")
        print(f"→ Intent: {result['intent']}")
        print(f"→ Category: {result['category']}")
        print(f"→ Matched: {result['match_utterance']}")
        print(f"→ Similarity: {result['similarity']}")
