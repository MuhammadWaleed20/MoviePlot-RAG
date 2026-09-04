import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
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
    db_path = os.path.join(current_dir, "..", "chroma_db")
    
    print("Initializing system (loading embeddings and database)...")
    # 1. Initialize Retriever & LLM ONLY ONCE
    retriever = get_retriever(db_path)
    
    # Note: Use whatever model string worked for you in the previous step
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 2. Design the Prompt Template
    template = """You are a movie expert assistant. Use the following retrieved movie plots to answer the question. 
    If you don't know the answer based on the context, just say that you don't know. Do not make up information.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    # 3. Build the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n✅ System Ready! Type 'exit' to quit.\n")
    
    # 4. Interactive Chat Loop
    while True:
        user_question = input("🗣️ Question: ")
        if user_question.lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
            
        print("🤖 Answer: ", end="", flush=True)
        
        # Use .stream() instead of .invoke() for instant word-by-word output
        for chunk in rag_chain.stream(user_question):
            print(chunk, end="", flush=True)
            
        print("\n" + "-"*50 + "\n")