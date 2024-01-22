from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    CHROMA_SETTINGS
)


embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})
# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# load the vectorstore
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS
)
retriever = db.as_retriever()

print(retriever.get_relevant_documents(query=""))