import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PythonLoader
from langchain.text_splitter import CharacterTextSplitter

pdf = os.path.abspath("./file.pdf")

if __name__ == '__main__':
    load_dotenv()

    loader = PythonLoader(pdf)
    document = loader.load()