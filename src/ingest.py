import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    # Initialize the LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Create LangChain Document objects containing the chunked text and metadata
    docs = text_splitter.create_documents(texts, metadatas=metadatas)
    
    print(f"Split data into {len(docs)} chunks!")
    return docs

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "data", "movie_synopsis.csv")
    
    # 1. Load data
    movie_texts, movie_metadata = load_and_clean_data(file_path)
    
    # 2. Chunk data
    document_chunks = chunk_data(movie_texts, movie_metadata)
    
    # Print the first chunk to inspect
    print("\n--- Sample Chunk 1 ---")
    print(document_chunks[0].page_content)
    print("--- Metadata ---")
    print(document_chunks[0].metadata)
    print("----------------------")