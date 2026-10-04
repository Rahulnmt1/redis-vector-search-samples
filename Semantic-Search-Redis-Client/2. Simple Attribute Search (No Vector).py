# Find all bikes with brand "Peaknetic"
# This purely uses metadata search to find bikes matching the brand "Peaknetic" without vector similarity.

"""
Example Query:

“Find all bikes by the brand ‘Peaknetic’.”

How it works:

Uses only metadata fields, exact matches (not semantic, just keyword/field matching).

Why it’s useful:

For filtering, product inventory, and structured queries.

"""
import redis
import json
from redis.commands.search.query import Query

# Connect to Redis (adjust host/port/password if needed)
client = redis.Redis(host="localhost", port=12000, decode_responses=True)

attribute_query = Query("@brand:Peaknetic").return_fields("id", "brand", "model", "price")

results = client.ft("idx:bikes_vss").search(attribute_query)

for doc in results.docs:
    print(f"Brand: {doc.brand}, Model: {doc.model}, Price: {doc.price}")

'''
Brand: Peaknetic, Model: Soothe Electric bike, Price: 1950
Brand: Peaknetic, Model: Secto, Price: 430
'''