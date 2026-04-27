import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader


def load_documents():
    docs = []

    
    with open("C:/Users/SWETHAKRISHNA/OneDrive/Desktop/Legal_Chat_Bot/docs/policy.txt", "r", encoding="utf-8") as f:
        docs.append(f.read())


    with open("C:/Users/SWETHAKRISHNA/OneDrive/Desktop/Legal_Chat_Bot/docs/rules.txt", "r", encoding="utf-8") as f:
        docs.append(f.read())

    return docs



def split_docs(docs, chunk_size=300):
    chunks = []
    for doc in docs:
        for i in range(0, len(doc), chunk_size):
            chunks.append(doc[i:i+chunk_size])
    return chunks



model = SentenceTransformer("all-MiniLM-L6-v2")
def create_faiss_index(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings



def retrieve(query, index, chunks, k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    results = [chunks[i] for i in indices[0]]
    return results


generator = pipeline("text-generation", model="distilgpt2")


def generate_answer(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)

    prompt = f"""
Answer the question based only on the context.

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(prompt, max_new_tokens=100)
    if isinstance(result, list) and len(result) > 0:
        if "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif "text" in result[0]:
            return result[0]["text"]

    return "No answer generated"


def main():
    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents...")
    chunks = split_docs(docs)

    print("Creating embeddings + FAISS index...")
    index, embeddings = create_faiss_index(chunks)

    print("Chatbot ready! Type 'exit' to quit.")

    while True:
        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            break

        results = retrieve(query, index, chunks)
        answer = generate_answer(query, results)

        print("\nAnswer:\n", answer)
        print("\nSources:")
        for r in results:
            print("-", r[:80], "...")  

if __name__ == "__main__":
    main()
