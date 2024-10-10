"""Index source documents and persist in vector embedding database."""

# Copyright (c) 2023 Brent Benson
#
# This file is part of [project-name], licensed under the MIT License.
# See the LICENSE file in this repository for details.

import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

from utils.embedding_store import generate_embed_index
from utils.embedding_store import pdf_to_chunks

# Load the environment variables from the .env file
load_dotenv()

SOURCE_DOCUMENTS = ["source_documents/5008_Federalist Papers.pdf"]

# Pre-load the tokenizer (move this outside if frequently calling the function)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def ingest_docs(source_documents):
    all_docs = []
    for source_doc in source_documents:
        print(source_doc)
        docs = pdf_to_chunks(source_doc, tokenizer=tokenizer)
        all_docs = all_docs + docs
    return all_docs


def main():
    print("Ingesting...")
    all_docs = ingest_docs(SOURCE_DOCUMENTS)
    print("Persisting...")
    db = generate_embed_index(all_docs)
    print("Done.")


if __name__ == "__main__":
    main()
