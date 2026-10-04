# Basic KNN Vector Search Query (Find top 3 most similar vectors)
# This performs a top-3 nearest neighbor search over all indexed bike embeddings using the cosine distance metric. The query vector is the embedding of your input text. The returned documents include metadata and a similarity score (converted from distance).

"""
Example Query:

“Show me the 3 bikes most similar to this query: ‘Best mountain bikes for kids’”

How it works:

The query is turned into a vector using the same embedding model.

Redis searches all bike embeddings for the 3 bikes whose description embeddings are mathematically closest (smallest cosine distance) to the query vector.

Why it’s useful:

Finds conceptually similar bikes even if the description has different words than the query.

For natural language search: you get results based on meaning, not exact words.

"""
import redis
import json
import numpy as np
from redis.commands.search.indexDefinition import IndexDefinition, IndexType   # ✅ correct for redis-py >=5.x
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# Connect to Redis (adjust host/port/password if needed)
client = redis.Redis(host="localhost", port=12000, decode_responses=True)

# Example natural language query
query_text = "Best Mountain bikes for kids"

# Encode query text to a vector embedding using your initialized embedder
embedder = SentenceTransformer("msmarco-distilbert-base-v4")
query_vector = embedder.encode([query_text])[0]

# Define KNN query to find 3 nearest neighbors based on vector similarity
knn_query = (
    Query("(*)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .dialect(2)
)

# Execute query - note usage of np.array for correct binary vector format
results = client.ft("idx:bikes_vss").search(
    knn_query,
    query_params={"query_vector": np.array(query_vector, dtype=np.float32).tobytes()}
)

# Print results with adjusted similarity score
for doc in results.docs:
    print(f"Brand: {doc.brand}, Model: {doc.model}, Similarity: {1 - float(doc.vector_score):.3f}")
    print(f"Description: {doc.description}\n")
