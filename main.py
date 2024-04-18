import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PythonLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

pdf = os.path.abspath("./file.pdf")

if __name__ == '__main__':
    load_dotenv()

    loader = PythonLoader(pdf)
    loader.encoding = "latin-1"
    document = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300, separator="\n")
    docs = splitter.split_documents(document)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

