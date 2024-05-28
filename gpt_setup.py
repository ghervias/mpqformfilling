import os
import json

import openai
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def mpq_to_dict():
    # Directory containing the JSON files
    directory_path = './mpq_files'

    # Initialize a dictionary to hold the content of all JSON files
    json_contents = {}
    # List all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                json_contents[filename] = json.load(file)
    print("returning MPQ dict...")
    return json_contents

def init_langchain(retriever):
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    chat_template = ChatPromptTemplate.from_template("""You are a search and rescue interview analyst assistant. \
            Your job is to extract important information from the given interview \
            transcript and use it to fill out the Missing Person Questionnaire, \
            which will also be given. \
            Question: {question} \
            Context: {context}""")

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | chat_template
        | llm
        | StrOutputParser()
    )
    return rag_chain
