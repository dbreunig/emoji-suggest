from sentence_transformers import SentenceTransformer, util
import numpy as np

emojis = []

with open('./emojis.txt', 'r') as file:
  for line in file:
    emojis.append(line.strip())

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer('clip-ViT-B-32')


embeddings = model.encode(emojis)

example = "making coffee with an aeropress"



# Generate embedding for the example string
example_embedding = model.encode(example, convert_to_tensor=True)

# Find the closest 5 sentences of the corpus for the example query based on cosine similarity
top_k = min(50, len(emojis))
cos_scores = util.pytorch_cos_sim(example_embedding, embeddings)[0]
cos_scores = cos_scores.cpu()

# We use np.argpartition, to only partially sort the top_k results
top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

print("\n\n======================\n\n")
print("Query:", example)
print("\nTop 5 most similar sentences in corpus:")

for idx in top_results[0:top_k]:
  print(emojis[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
