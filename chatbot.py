import chromadb
from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from dotenv import load_dotenv
import markdown
from chromadb.config import Settings
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configuration
load_dotenv()
API_KEY = os.getenv("API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
QUERY_MODEL = os.getenv("QUERY_MODEL")

# Initialize Flask app
app = Flask(__name__)

# Initialize Chroma client
chroma_client = chromadb.Client(Settings(persist_directory="./Chroma_store"))
genai.configure(api_key=API_KEY)
chat_history = []

# Load preprocessed documents and embeddings into memory
with open("all_texts.pkl", 'rb') as f:
    all_texts = pickle.load(f)

with open("PKG02_embeddings.pkl", 'rb') as f:
    combined_embeddings = pickle.load(f)

def create_embeddings(text_list):
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text_list
    )
    return response['embedding']

def retrieve_relevant_documents(query, document_embeddings, top_k=20):
    query_embedding_response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=[query]
    )
    query_embedding = query_embedding_response['embedding'][0]
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    return [all_texts[i] for i in sorted_indices[:top_k]], similarities[sorted_indices[:top_k]]

def query_documents_with_gemini(query, context):
    prompt = f"""
    You are a helpful assistant. Answer the query below based on the provided context in extensive, informative and professional manner:

    Query: {query}

    Context:
    {context}

    Assistant:
    """
    model = genai.GenerativeModel(QUERY_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

async def process_query(user_query):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        relevant_docs, _ = await loop.run_in_executor(pool, retrieve_relevant_documents, user_query, combined_embeddings)
        context = "\n".join(relevant_docs[:5])  # Limit to the top 5 most relevant documents for efficiency
        final_answer = await loop.run_in_executor(pool, query_documents_with_gemini, user_query, context)
    return final_answer

@app.route("/", methods=["GET", "POST"])
async def chat():
    if request.method == "POST":
        user_query = request.form.get("query")
        response = await process_query(user_query)
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": response})
        return jsonify({"response": markdown.markdown(response)})
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)
