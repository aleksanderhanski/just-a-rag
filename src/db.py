from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os

# shared config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "documents"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768
DISTANCE = Distance.COSINE

client = QdrantClient(url=QDRANT_URL)


def ensure_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )


def reset_collection():
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
    )