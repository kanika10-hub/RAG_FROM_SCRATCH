from huggingface_hub import InferenceClient
import os
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
import faiss
##client = InferenceClient(model="mistral-7b-instruct-v0.1.Q4_0.gguf")
##embeddings = client.embeddings(input=["Hello world!"])
##print(embeddings)
# Load the environment variables from the .env file
load_dotenv()
from huggingface_hub import InferenceClient
import huggingface_hub

print("Hugging Face Hub Version:", huggingface_hub.__version__)
hug_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
model_id = os.getenv("HUGGINGFACEHUB_MODEL_ID")
print(f"Using model: {model_id}")
print(f"Using token: {hug_token}")

client = InferenceClient(token=hug_token, model="sentence-transformers/all-MiniLM-L6-v2")

#pdf reader 
reader = PdfReader("document.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

#create chunks 

chunk_size = 500

documents = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

#total chunks 
print("Total Chunks:", len(documents))

document_embeddings = []

for doc in documents:
    embedding = client.feature_extraction(doc)
    document_embeddings.append(embedding)

print("Total embeddings:", len(document_embeddings))

##for faiss
embeddings_np = np.array(
    document_embeddings,
    dtype=np.float32
)

print("Embedding Shape:", embeddings_np.shape)

dimension = embeddings_np.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings_np)
print("Vectors Stored:", index.ntotal)

# --------------------
# User Query
# --------------------

query = input("Ask a question about the PDF: ")

query_embedding = client.feature_extraction(query)

query_np = np.array(
    [query_embedding],
    dtype=np.float32
)

# --------------------
# Cosine Similarity Function
# --------------------

#def cosine_similarity(vec1, vec2):

 #   vec1 = np.array(vec1)
 #   vec2 = np.array(vec2)

 #   similarity = np.dot(vec1, vec2) / (
   #     np.linalg.norm(vec1) * np.linalg.norm(vec2)
 #   )

 #   return similarity
#--------------------
# Compare Query with Documents
# --------------------

#scores = []

#for emb in document_embeddings:

   # score = cosine_similarity(query_embedding, emb)

  #  scores.append(score)

# --------------------
# Display Scores
# --------------------

#print("\nSimilarity Scores:")

#for i, score in enumerate(scores):
   # print(f"Document {i+1}: {score:.4f}")

# --------------------
# Best Match
# --------------------


top_k = 2

distances, indices = index.search(
    query_np,
    top_k
)

#sorted_indices = np.argsort(scores)[::-1]


print("\nTop Documents:")
retrieved_docs = []
for idx in indices[0]:
   # print()
   # print("Score:", scores[idx])
    print("Document:", documents[idx])
    retrieved_docs.append(documents[idx])


print("\nRetrieved Chunks:")

for idx in indices[0]:
    print("-" * 50)
    print(documents[idx][:300])
print("\nRetrieved Documents:")
for doc in retrieved_docs:
    print(doc)

#create context
context = "\n".join(retrieved_docs)

print("\nContext:")
print(context)

#build prompt

prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

print(prompt)

#Send to an LLM
#Change your client to an instruction/chat model:

llm_client = InferenceClient(
    token=hug_token,
    model="meta-llama/Llama-3.1-8B-Instruct"
)

response = llm_client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=200
)

print("\nAnswer:")
print(response.choices[0].message.content)
