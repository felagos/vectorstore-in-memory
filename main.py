import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PythonLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI

pdf = os.path.abspath("./file.pdf")

if __name__ == '__main__':
    load_dotenv()

    loader = PythonLoader(pdf)
    loader.encoding = "latin-1"
    document = loader.load()

    splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=45, separator="\n")
    docs = splitter.split_documents(document)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index_vector")

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    query = "Give me the gist of ReAct in 3 sentences"
    res = qa.run(query)
    print(res)
