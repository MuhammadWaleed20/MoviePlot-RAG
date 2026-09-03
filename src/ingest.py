import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_and_clean_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['title', 'synopsis'])
    df['rich_text'] = "Movie Title: " + df['title'] + "\nPlot: " + df['synopsis']
    
    texts = df['rich_text'].tolist()
    metadatas = [{"title": title} for title in df['title'].tolist()]
    
    print(f"Successfully loaded and cleaned {len(texts)} movies!")
    return texts, metadatas

def chunk_data(texts, metadatas):
    print("Chunking text data...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.create_documents(texts, metadatas=metadatas)
    print(f"Split data into {len(docs)} chunks!")
    return docs

def create_vector_db(docs, persist_dir):
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building and persisting Chroma vector database (this may take a minute)...")
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=persist_dir
    )
    print(f"Successfully saved vector database to {persist_dir}!")
    return vector_store

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "data", "movie_synopsis.csv")

    db_path = os.path.join(current_dir, "..", "chroma_db")

    movie_texts, movie_metadata = load_and_clean_data(file_path)

    document_chunks = chunk_data(movie_texts, movie_metadata)
 
    create_vector_db(document_chunks, db_path)