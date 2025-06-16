import os
from dotenv import load_dotenv

# --- Core LangChain components ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
# --- NEW: Import PromptTemplate ---
from langchain.prompts import PromptTemplate

def main():
    """
    Main function to set up and run the QA chatbot.
    """
    # --- 1. Load Environment and Configuration ---
    print("Loading environment variables...")
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found. Please add it to your .env file.")
        return

    DB_DIR = "chroma_db"
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
    LLM_MODEL_NAME = 'llama-3.3-70b-versatile'

    # --- 2. Load the Knowledge Base (Vector DB) ---
    print("Loading knowledge base from ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever()
    print("Knowledge base loaded successfully.")

    # --- 3. Initialize the Large Language Model (LLM) ---
    print(f"Initializing LLM: {LLM_MODEL_NAME}...")
    llm = ChatGroq(model_name=LLM_MODEL_NAME)
    print("LLM initialized.")
    
    # --- NEW: Define the Custom Prompt Template ---
    # This template instructs the LLM to act as a professor and synthesize answers.
    prompt_template = """
You are an expert professor specializing in astrophysics, robotics, and space exploration. Your tone is knowledgeable, confident, and clear.

Use the following retrieved context to formulate your answer. Synthesize the information and present it as your own knowledge.

Do NOT mention that you are answering based on a provided context or documents. Avoid phrases like 'According to the document,' 'The text states,' or 'Based on the provided information.'

If the context does not contain the answer, state that the topic is outside your current specialized knowledge base without mentioning the documents.

CONTEXT:
{context}

QUESTION: {question}

PROFESSORIAL ANSWER:
"""

    # Create a PromptTemplate object
    PROFESSOR_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # --- 4. Create the Retrieval-Augmented Generation (RAG) Chain ---
    # --- MODIFIED: Inject the custom prompt ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # We inject our custom prompt here
        chain_type_kwargs={"prompt": PROFESSOR_PROMPT}
    )
    print("QA chain with custom 'Professor' prompt created. The chatbot is ready.")

    # --- 5. Start the Interactive Command-Line Interface (CLI) ---
    print("\n--- NASA RAG Chatbot (Professor Mode) ---")
    print("Ask a question about the documents. Type 'exit' to quit.")

    while True:
        try:
            query = input("\nYour question: ")
            if query.lower() == 'exit':
                print("Exiting chatbot. Goodbye!")
                break
            
            if not query.strip():
                continue

            print("\nThinking...")
            
            result = qa_chain.invoke({"query": query})
            
            print("\nAnswer:")
            print(result["result"])

        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot. Goodbye!")
            break

if __name__ == "__main__":
    main()