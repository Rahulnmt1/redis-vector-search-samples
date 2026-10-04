# This query returns documents where the vector distance from the query vector is less than or equal to 0.55. The documents are sorted by distance (lower is better).

"""
Example Query:

“Show me all bikes that are sufficiently similar (distance ≤ 0.55) to this query: ‘Comfortable city bike’.”

How it works:

Like KNN, but instead of a fixed number of results, this returns all bikes whose vector distance to the query is within the set threshold.

Why it’s useful:

Useful for finding all bikes that are “close enough” rather than just the top few (e.g., “Show me all potential commuter bikes,” not just the best one).

"""
# This query returns documents where the vector distance from the query vector is less than or equal to 0.55. The documents are sorted by distance (lower is better).

import redis
import json
import numpy as np
from redis.commands.search.indexDefinition import IndexDefinition, IndexType   # ✅ correct for redis-py >=5.x
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# Connect to Redis (adjust host/port/password if needed)
client = redis.Redis(host="localhost", port=12000, decode_responses=True)

embedder = SentenceTransformer("msmarco-distilbert-base-v4")
query_text = "Best mountain bikes for kids"
query_vector = embedder.encode([query_text])

range_threshold = 0.55  # maximum vector distance allowed

range_query = (
    Query("@vector:[VECTOR_RANGE $range $query_vector]=>{$YIELD_DISTANCE_AS:vector_score}")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .paging(0, 5)  # get top 5 results
    .dialect(2)
)

results = client.ft("idx:bikes_vss").search(
    range_query,
    query_params={
        "range": range_threshold,
        "query_vector": np.array(query_vector, dtype=np.float32).tobytes()
    }
)

for doc in results.docs:
    print(f"Brand: {doc.brand}, Model: {doc.model}, Distance: {float(doc.vector_score):.3f}")
