import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


import os
import openai
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions


def load_transcript(text):
    # loader = TextLoader('/content/transcript_1.txt')
    # documents = loader.load()
    # print(documents)
    #
    # with open("/content/transcript_1.txt") as f:
    #     transcript_1 = f.read()
    # load_dotenv()
    # openai.api_key = os.getenv('OPENAI_API_KEY')
    #
    # text_splitter = SemanticChunker(OpenAIEmbeddings())
    # chunks = text_splitter.create_documents([text])
    # print(chunks)
    #
    # client = weaviate.Client(
    #     embedded_options=EmbeddedOptions()
    # )
    # print("Creating vectorstore...")
    # vectorstore = Weaviate.from_documents(
    #     client=client,
    #     documents=chunks,
    #     embedding=OpenAIEmbeddings(),
    #     by_text=False
    # )
    # print("Creating retriever...")
    # retriever = vectorstore.as_retriever(
    #     search_kwargs={
    #         "k": 5
    #     }
    # )

    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()
    return retriever