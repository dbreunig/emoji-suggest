from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import numpy as np

#
# Load the model
model = SentenceTransformer('clip-ViT-B-32')

#
# Load the emojis & encode
emojis = []
with open('./steplist_emojis.txt', 'r') as file:
  for line in file:
    emojis.append(line.strip())
embeddings = model.encode(emojis)

#
# Create the FastAPI app
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("healthcheck")
def read_root():
    return {"status": "ok"}

@app.get("/{text}")
def get_emoji(text: str):
    # Encode the input text
    text_embedding = model.encode([text])

    # Calculate cosine similarity between input text embedding and emoji embeddings
    similarities = util.cos_sim(text_embedding, embeddings)

    # Find the index of the most similar emoji
    most_similar_index = np.argmax(similarities)

    # Return the most similar emoji
    return {"emoji": emojis[most_similar_index]}
