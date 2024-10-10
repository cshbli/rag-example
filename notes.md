## Create a python virtual environment
```
conda create -n rag python=3.11
conda activate rag
```

- Using Chroma vector store
```
## Chroma vector store
CHROMA_PERSIST_DIR=doc_index
```

- After install the requirements, install `langchain-community`
```
pip install -U langchain-community
# To use OpenAIEmbeddings
pip install tiktoken
```

4. Run the document indexing script - `python index_documents.py`
5. Run a sample of searching the index - `python search_index.py`
6. Run a sample of searching the index in a web browser - `streamlit run search_index_ui.py`

- Using OpenAI model
```
## OpenAI
OPENAI_API_KEY="your OpenAI API key"
```
