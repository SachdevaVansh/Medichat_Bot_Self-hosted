from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List 

def create_faiss_index(documents):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(documents,embeddings)

def retrieve_relevant_docs(vectorstore: FAISS, query:str, k:int=4):
    return vectorstore.similarity_search(query,k=k)
