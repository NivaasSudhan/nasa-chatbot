import os
import requests
import yaml
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_pdf_urls(config: Dict) -> List[Dict]:
    """
    Extract PDF URLs and names from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of dictionaries containing PDF URLs and names
    """
    return [{'url': item['url'], 'name': item['name']} for item in config['pdf_urls']]

def download_pdfs(pdf_entries, save_dir='pdfs'):
    """
    Downloads PDFs from their URLs and saves them to a directory.
    
    Args:
        pdf_entries: List of dictionaries containing 'url' and 'name' for each PDF
        save_dir: Directory to save PDFs
    """
    print(f"Creating directory '{save_dir}' to store PDFs...")
    os.makedirs(save_dir, exist_ok=True)
    
    for entry in pdf_entries:
        url = entry['url']
        name = entry['name']
        
        # Create a safe filename from the book name
        safe_filename = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = os.path.join(save_dir, f"{safe_filename}.pdf")
        
        if os.path.exists(filename):
            print(f"File '{name}' already exists at {filename}. Skipping download.")
            continue

        try:
            print(f"Downloading '{name}' from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded '{name}' to {filename}")
        
        except requests.exceptions.RequestException as e:
            print(f"Failed to download '{name}'. Error: {e}")

def process_and_store_documents(pdf_dir='pdfs', db_dir='chroma_db'):
    """
    Loads documents from a directory, processes them, and stores them in a vector database.
    """
    print(f"\nLoading documents from '{pdf_dir}'...")
    # This loader will find and load all PDFs in the specified directory.
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()
    if not documents:
        print("No documents were loaded. Please check the PDF directory.")
        return

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into smaller chunks...")
    # This splitter will break documents into chunks of 1000 characters, with a 200 character overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    print("Loading sentence-transformer model for embeddings...")
    # We use a popular, efficient model from Hugging Face for creating the embeddings.
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    print(f"Creating and persisting vector store in '{db_dir}'...")
    # This command creates the embeddings and stores them in the Chroma vector store.
    # It will be saved to the 'chroma_db' directory for future use.
    db = Chroma.from_documents(texts, embeddings, persist_directory=db_dir)
    
    print("\n--- Data ingestion complete! ---")
    print(f"The knowledge base is stored in the '{db_dir}' directory.")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Get URLs and directories from config
    PDF_ENTRIES = get_pdf_urls(config)
    PDF_SAVE_DIRECTORY = config['directories']['pdf_save_dir']
    VECTOR_DB_DIRECTORY = config['directories']['vector_db_dir']

    # Run the download process
    # download_pdfs(PDF_ENTRIES, PDF_SAVE_DIRECTORY)
    
    # Run the ingestion process
    process_and_store_documents(PDF_SAVE_DIRECTORY, VECTOR_DB_DIRECTORY)