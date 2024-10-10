"""Utils for vector embedding database."""

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.vectorstores import PGVector

COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Function to create a Chroma vector store and persist it
def create_index_chroma(docs, embeddings, persist_dir):
    # Create the Chroma vector store from the documents and their embeddings
    db = Chroma.from_documents(
        documents=docs,              # The list of documents to embed
        embedding=embeddings,        # The Hugging Face embedding model
        collection_name=COLLECTION_NAME,  # Collection name for organizing documents
        persist_directory=persist_dir  # Directory to persist the vector store on disk
    )
    
    # Since Chroma automatically persists, manual persistence is not needed anymore
    # db.persist()  # (Deprecated) Call to persist manually - not necessary
    return db  # Return the Chroma vector store


def get_chroma_db(embeddings, persist_dir):
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db


# Function to create an OpenSearch vector store
def create_index_opensearch(docs, embeddings, url):
    # Retrieve OpenSearch credentials from environment variables
    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")
    
    # Create the OpenSearch vector store from the documents and their embeddings
    db = OpenSearchVectorSearch.from_documents(
        docs,                        # The list of documents to embed
        embeddings,                  # The Hugging Face embedding model
        index_name=COLLECTION_NAME,  # The index name in OpenSearch
        opensearch_url=url,          # URL to the OpenSearch cluster
        http_auth=(username, password),  # Authentication credentials for OpenSearch
        use_ssl=False,               # Whether to use SSL (set to False for simplicity)
        verify_certs=False,          # Disable certificate verification
        ssl_assert_hostname=False,   # Skip hostname verification
        ssl_show_warn=False          # Disable SSL warning messages
    )
    return db  # Return the OpenSearch vector store


def get_opensearch_db(embeddings, url):
    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")
    db = OpenSearchVectorSearch(
        embedding_function=embeddings,
        index_name=COLLECTION_NAME,
        opensearch_url=url,
        http_auth=(username, password),
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    return db


# Function to create a Postgres vector store using PGVector
def create_index_postgres(docs, embeddings, connection_string):
    # Create the Postgres vector store using PGVector from the documents and embeddings
    db = PGVector.from_documents(
        docs,       # The list of documents to index
        embeddings, # The embeddings object to use for encoding the documents
        collection_name=COLLECTION_NAME,        # The name of the collection to store the documents
        connection_string=connection_string,    # The connection string for the Postgres database
    )
    return db


def get_postgres_db(embeddings, connection_string):
    db = PGVector(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=connection_string,
    )
    return db


# Function to generate an embedding index based on available environment variables
# The function checks environment variables to decide which vector store (Chroma, OpenSearch, Postgres) to use.
def generate_embed_index(docs):
    # Initialize Hugging Face embeddings with a specified model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Fetch environment variables for vector stores
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    opensearch_url = os.getenv("OPENSEARCH_URL")
    postgres_conn = os.getenv("POSTGRES_CONNECTION")
    
    # Check if Chroma is configured (using persistence directory)
    if chroma_persist_dir:
        # Create Chroma index
        db = create_index_chroma(docs, embeddings, chroma_persist_dir)
    # If Chroma isn't configured, check for OpenSearch
    elif opensearch_url:
        # Create OpenSearch index
        db = create_index_opensearch(docs, embeddings, opensearch_url)
    # If neither Chroma nor OpenSearch is configured, check for Postgres
    elif postgres_conn:
        # Create Postgres index using PGVector
        db = create_index_postgres(docs, embeddings, postgres_conn)
    else:
        # You can add additional vector stores here

        # If no vector store is configured, raise an error
        raise EnvironmentError("No vector store environment variables found.")
    
    # Return the created vector store
    return db


def get_embed_db(embeddings):
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    opensearch_url = os.getenv("OPENSEARCH_URL")
    postgres_conn = os.getenv("POSTGRES_CONNECTION")
    if chroma_persist_dir:
        db = get_chroma_db(embeddings, chroma_persist_dir)
    elif opensearch_url:
        db = get_opensearch_db(embeddings, opensearch_url)
    elif postgres_conn:
        db = get_postgres_db(embeddings, postgres_conn)
    else:
        # You can add additional vector stores here
        raise EnvironmentError("No vector store environment variables found.")
    return db


def pdf_to_chunks(pdf_file, tokenizer):
    """
    Converts a PDF into chunks of text using a tokenizer and text splitter.
    Tokenizer is loaded only once and passed as an argument for optimization.
    """
    # Use RecursiveCharacterTextSplitter with the Hugging Face tokenizer
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n \n", "\n\n", "\n", " ", ""],  # Customize separators as needed
        chunk_size=512,  # Adjust chunk size according to embedding model's needs
        chunk_overlap=0
    )

    # Use PyPDFLoader to load the PDF and split the content into chunks
    loader = PyPDFLoader(pdf_file)

    # Load and split the document into chunks
    docs = loader.load_and_split(text_splitter)

    return docs

