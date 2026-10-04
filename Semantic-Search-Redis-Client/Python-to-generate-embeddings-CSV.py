import numpy as np
import csv

# Simulated full length vectors - using random floats to illustrate
np.random.seed(0)
query_vector = np.random.uniform(-0.1, 0.1, 768).astype(np.float32)
bike_vector = np.random.uniform(-0.1, 0.1, 768).astype(np.float32)

# Write to CSV for easy inspection
filename = "full_query_and_bike_embeddings.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["dimension_index", "query_embedding", "bike_embedding"])
    for i in range(len(query_vector)):
        writer.writerow([i+1, query_vector[i], bike_vector[i]])

filename