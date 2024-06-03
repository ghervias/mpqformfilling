import os
from dotenv import load_dotenv
import json
import re

from flask import Flask, request, jsonify, current_app
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings


ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.json.sort_keys = False
openai.api_key = 'your_openai_api_key'


@app.route('/')
def index():
    return current_app.send_static_file('homepage.html')


@app.route('/process', methods=['POST'])
def process_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        text = file.read().decode('utf-8')

        retriever = load_transcript(text)
        mpq_dict = mpq_to_dict()
        rag_chain = init_langchain(retriever)
        response, trio_list = run_query(rag_chain, mpq_dict, retriever)
        print(response)
        print(jsonify(response))
        return jsonify(response)
    return "Invalid file type."


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_transcript(text):
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()
    return retriever

def mpq_to_dict():
    directory_path = './mpq_files'

    json_contents = {}
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

def replace_patterns(input_string):
    # Define the patterns and replacements
    patterns_replacements = {
        r"': \{'": r'": {"',
        r"'}, '": r'"}, "',
        r"': '": r'": "',
        r"', '": r'", "',
        r"'}}": r'"}}',
        r"\{'": r'{"',
        r"\\'s ": r"'s",
        r"'}": r'"}',
        r"':": r'":',
        r", '": r', "'
    }

    # Apply each pattern replacement
    for pattern, replacement in patterns_replacements.items():
        input_string = re.sub(pattern, replacement, input_string)

    return input_string


def run_query(rag_chain, mpq_dict, retriever):
    query = "Provide a  RFC8259 compliant JSON response following this format without \
    deviation. If there is no relevant information in the transcript, keep the original \
    JSON structure, but fill in the field with 'N/A'. Important: DO NOT provide any \
    information in a field that is not directly relevant. Here is the form: "

    # Function to retrieve context
    def retrieve_context(query):
        return retriever.get_relevant_documents(query)

    trio_list = []
    print("running query...")
    final_json = {}
    for item in mpq_dict.keys():
        if(item != "CognitivelyImpairedandIntellectualDisability.json"):
            continue
        print(item)
        response = rag_chain.invoke(query + str(mpq_dict[item]))
        context_documents = retriever.get_relevant_documents(query + str(mpq_dict[item]))
        context = [doc.page_content for doc in context_documents]
        response_json = json.loads(replace_patterns(response))
        final_json[item.removesuffix('.json')] = response_json

        # for field_name, field_value in response_json.items():
        #     if (field_value == "N/A" or isinstance(field_value, dict)):
        #         continue
        #     trio_list.append([mpq_dict[item][field_name], field_value, context])

    final_json = json.loads(json.dumps(final_json, indent=4))
    return final_json, trio_list


if __name__ == '__main__':
    app.run(debug=True)
