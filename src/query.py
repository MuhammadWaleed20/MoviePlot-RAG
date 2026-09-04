import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the API key from the .env file
load_dotenv()

def get_retriever(db_path):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    # Combines the retrieved chunks into a single text block for the LLM to read
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "..", "chroma_db")
    
    print("Initializing system (loading embeddings and database)...")
    # 1. Initialize Retriever & LLM ONLY ONCE
    retriever = get_retriever(db_path)
    
    # Note: Use whatever model string worked for you in the previous step
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    
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