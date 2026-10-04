"""
Code samples for vector database quickstart pages:
    https://redis.io/docs/latest/develop/get-started/vector-database/
"""
import json
import numpy as np
import pandas as pd
import requests
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType   # ✅ correct for redis-py >=5.x
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# 1. Fetch dataset
URL = "https://raw.githubusercontent.com/bsbodden/redis_vss_getting_started/main/data/bikes.json"
response = requests.get(URL, timeout=10)
bikes = response.json()

# 2. Connect to Redis on port 12000
client = redis.Redis(host="localhost", port=12000, decode_responses=True)
assert client.ping()

# 3. Store demo data in Redis
pipeline = client.pipeline()
for i, bike in enumerate(bikes, start=1):
    redis_key = f"bikes:{i:03}"
    pipeline.json().set(redis_key, "$", bike)
pipeline.execute()

# 4. Generate text embeddings for descriptions
keys = sorted(client.keys("bikes:*"))
descriptions = client.json().mget(keys, "$.description")
descriptions = [item for sublist in descriptions for item in sublist]

embedder = SentenceTransformer("msmarco-distilbert-base-v4")
embeddings = embedder.encode(descriptions).astype(np.float32).tolist()
VECTOR_DIMENSION = len(embeddings[0])  # 768

# Insert embeddings into Redis
pipeline = client.pipeline()
for key, embedding in zip(keys, embeddings):
    pipeline.json().set(key, "$.description_embeddings", embedding)
pipeline.execute()

# 5. Create index
schema = (
    TextField("$.model", no_stem=True, as_name="model"),
    TextField("$.brand", no_stem=True, as_name="brand"),
    NumericField("$.price", as_name="price"),
    TagField("$.type", as_name="type"),
    TextField("$.description", as_name="description"),
    VectorField("$.description_embeddings", "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIMENSION,
        "DISTANCE_METRIC": "COSINE",
    }, as_name="vector"),
)
definition = IndexDefinition(prefix=["bikes:"], index_type=IndexType.JSON)
client.ft("idx:bikes_vss").create_index(fields=schema, definition=definition)

# 6. Perform vector search
queries = [
    "Bike for small kids",
    "Best Mountain bikes for kids",
    "Cheap Mountain bike for kids", 
    "Female specific mountain bike",
    "Road bike for beginners"
]
encoded_queries = embedder.encode(queries)

query = (
    Query("(*)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .dialect(2)
)

results = []
for i, encoded_query in enumerate(encoded_queries):
    result_docs = client.ft("idx:bikes_vss").search(
        query,
        {"query_vector": np.array(encoded_query, dtype=np.float32).tobytes()},
    ).docs
    for doc in result_docs:
        vector_score = round(1 - float(doc.vector_score), 2)
        results.append({
            "query": queries[i],
            "score": vector_score,
            "id": doc.id,
            "brand": doc.brand,
            "model": doc.model,
            "description": doc.description,
        })

# Display as table
queries_table = pd.DataFrame(results)
print(queries_table.to_markdown(index=False))