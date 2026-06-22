from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

#Initializing Qdrant client using Docker container
client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)