import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "..", "vector_db")

def ask_question(query_text):
    print(f"🔍 Loading vector database from: {VECTOR_DB_DIR}")
    
    if not os.path.exists(VECTOR_DB_DIR):
        print("❌ Vector database not found! Did you run ingest.py first?")
        return

    # 1. Load the same embedding model used during ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Load the local FAISS database
    vector_db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    
    print(f"\n❓ Question: '{query_text}'\n")
    
    # 3. Perform similarity search (retrieve top 3 matching chunks)
    docs = vector_db.similarity_search(query_text, k=3)
    
    print("🎯 --- RETRIEVED CONTEXT FROM PAPER ---")
    for i, doc in enumerate(docs):
        print(f"\n[Match {i+1}]")
        print(doc.page_content)
    print("----------------------------------------")

if __name__ == "__main__":
    # Test our query system with a question about the Attention paper
    test_query = "What is the Transformer architecture based on?"
    ask_question(test_query)