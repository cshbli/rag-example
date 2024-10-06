"""Simplest script for creating retrieval pipeline and invoking an LLM."""

# Copyright (c) 2023 Brent Benson
#
# This file is part of [project-name], licensed under the MIT License.
# See the LICENSE file in this repository for details.

import os
import pprint
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import AzureChatOpenAI, BedrockChat
from langchain.vectorstores import Chroma
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.vectorstores.pgvector import PGVector
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAI

# Log full text sent to LLM
VERBOSE = False

# Details of persisted embedding store index
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Size of window for buffered window memory
MEMORY_WINDOW_SIZE = 10


def main():
    # Check which environment variables are set and use the appropriate LLM
    openai_model_name = os.getenv("OPENAI_MODEL_NAME")
    aws_credential_profile_name = os.getenv("AWS_CREDENTIAL_PROFILE_NAME")
    aws_bedrock_model_name = os.getenv("AWS_BEDROCK_MODEL_NAME")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Access persisted embeddings and expose through langchain retriever
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = get_embed_db(embeddings)
    retriever = db.as_retriever()

    if openai_key:
        print("Using OpenAI for language model.")
        llm = OpenAI(openai_api_key=openai_key)
    elif openai_model_name:
        print("Using Azure for language model.")
        llm = AzureChatOpenAI(
            temperature=0.5, deployment_name=openai_model_name, verbose=VERBOSE
        )
    elif aws_credential_profile_name and aws_bedrock_model_name:
        print("Using Amazon Bedrock for language model.")
        llm = BedrockChat(
            credentials_profile_name=aws_credential_profile_name,
            model_id=aws_bedrock_model_name,
            verbose=VERBOSE,
        )
    else:
        # One could add additional LLMs here
        raise EnvironmentError("No language model environment variables found.")

    # Establish a memory buffer for conversational continuity
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        window_size=MEMORY_WINDOW_SIZE,
    )

    # Put together all of the components into the full
    # chain with memory and retrieval-agumented generation
    query_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        verbose=VERBOSE,
        return_source_documents=True,
    )

    prompt = (
        "How should government responsibility be divided between "
        "the states and the federal government?"
    )
    query_response = query_chain({"question": prompt})
    pprint.pprint(query_response)


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


def get_chroma_db(embeddings, persist_dir):
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db


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


def get_postgres_db(embeddings, connection_string):
    db = PGVector(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=connection_string,
    )
    return db


if __name__ == "__main__":
    main()
