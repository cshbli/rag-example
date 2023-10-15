"""Index source documents and persist in vector embedding database."""

# Copyright (c) 2023 Brent Benson
#
# This file is part of [project-name], licensed under the MIT License.
# See the LICENSE file in this repository for details.

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

SOURCE_DOCUMENT = "source_documents/5008_Federalist Papers.pdf"
COLLECTION_NAME = "federalist_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "doc_index"


def main():
    docs = pdf_to_chunks(SOURCE_DOCUMENT)
    db = generate_embed_index(docs, COLLECTION_NAME, PERSIST_DIR)
    db.persist()


def pdf_to_chunks(pdf_file):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split()
    return docs


def generate_embed_index(docs, collection_name, persist_dir):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return db


if __name__ == "__main__":
    main()
