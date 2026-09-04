import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever(db_path):
    print("Loading embedding model...")
    # We MUST use the exact same embedding model we used for ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Connecting to ChromaDB...")
    # Load the existing database from the directory
    vector_store = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )
    
    # Convert the database into a retriever object
    # search_kwargs={"k": 3} tells it to return the top 3 most relevant plot chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

if __name__ == "__main__":
    # Point to the existing database folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "..", "chroma_db")
    
    # Initialize the retriever
    retriever = get_retriever(db_path)
    
    # Test query - feel free to change this to test your database!
    test_query = "space technology and time travel."
    print(f"\n🔍 Searching for: '{test_query}'\n")
    
    # Fetch the relevant chunks
    results = retriever.invoke(test_query)
    
    # Display the results
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} | Movie: {doc.metadata.get('title', 'Unknown')} ---")
        print(doc.page_content)
        print("-" * 50 + "\n")