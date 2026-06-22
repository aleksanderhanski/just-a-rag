import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")

# safe also when collection does not exist
client.delete_collection(collection_name="demo_collection")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

chunks = [
    "Qdrant is a vector database written in Rust.",
    "RAG combines retrieval with a language model to ground answers.",
    "Cosine distance pairs well with normalized embeddings.",
]

vectors = model.encode(chunks, normalize_embeddings=True)

points = [
    PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk)),
        vector=vec.tolist(),
        payload={"text": chunk},
    )
    for chunk, vec in zip(chunks, vectors)
]

client.upsert(collection_name="demo_collection", points=points)
print("Upserted:", len(points))
print("Count in collection:", client.count("demo_collection").count)


query = "What is Qdrant?"

# bge-base needs this query prefix
query_prefix = "Represent this sentence for searching relevant passages: "
query_vector = model.encode(query_prefix + query, normalize_embeddings=True).tolist()

results = client.query_points(
    collection_name="demo_collection",
    query=query_vector,
    limit=3,
    with_payload=True,
)

print(f"\nResults for: {query!r}\n")
for rank, point in enumerate(results.points, 1):
    print(f"{rank}. [{point.score:.3f}] {point.payload['text']}")