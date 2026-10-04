# Filter by brand "Peaknetic", then search similar vectors
# This query restricts vector similarity search to documents where the brand is "Peaknetic," combining semantic search with metadata filtering.

""""
Example Query:

“Find the 3 bikes most similar to this query, but only among bikes from the brand ‘Peaknetic’.”

How it works:

First, narrow to all documents where brand is Peaknetic.

Then, do the same vector search as above among that filtered set.

Why it’s useful:

Let’s you do semantic search within selected product lines, categories, or brands.

Highly relevant for narrowing down big catalogs while still getting smart matches.

"""

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
query_vector = embedder.encode([query_text])[0]

hybrid_query = (
    Query("(@brand:Peaknetic)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .dialect(2)
)

results = client.ft("idx:bikes_vss").search(
    hybrid_query,
    query_params={"query_vector": np.array(query_vector, dtype=np.float32).tobytes()}
)

for doc in results.docs:
    print(f"Brand: {doc.brand}, Model: {doc.model}, Similarity: {1 - float(doc.vector_score):.3f}")

'''
Brand: Peaknetic, Model: Soothe Electric bike, Similarity: 0.139
Brand: Peaknetic, Model: Secto, Similarity: 0.084
'''