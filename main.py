import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PythonLoader

pdf = os.path.abspath("./file.pdf")

if __name__ == '__main__':
    load_dotenv()

    loader = PythonLoader(pdf)