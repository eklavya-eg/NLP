from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from use import TensorflowHubEmbeddings


def create_vector_db(Data, chunk_size, chunk_overlap, path):
    loader = DirectoryLoader(Data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                         model_kwargs={"device": "cpu"})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(path)

if __name__=="__main__":
    DB_FAISS_PATH = "vectorstores/db_faiss1"
    try:
        os.makedirs("vectorstores/db_faiss1")
    except FileExistsError:
        pass
    data = "pdfs"
    create_vector_db(data, 100, 40, DB_FAISS_PATH)