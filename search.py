"""Search the indexed documents in Qdrant.

    python search.py "your question here"
"""

import sys

from sentence_transformers import SentenceTransformer

from db import client, COLLECTION_NAME, MODEL_NAME

# bge-base model needs query prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def search(query, limit=5):
    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    print(f"\nResults for: {query!r}\n")
    for rank, point in enumerate(results.points, 1):
        p = point.payload
        snippet = p["text"][:160].replace("\n", " ")
        print(f"{rank}. [{point.score:.3f}] {p['source']} - page {p['page']}, chunk {p['chunk']}")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is this document about?"
    search(query)