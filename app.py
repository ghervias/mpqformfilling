from flask import Flask, request, jsonify
import openai
from transcript_loader import load_transcript
from gpt_setup import mpq_to_dict, init_langchain
from run_query import run_query
from run_ragas import ragas_eval
import langchain  # Make sure you import the necessary parts of langchain

app = Flask(__name__)
app.json.sort_keys = False


# Configure your OpenAI API key
openai.api_key = 'your_openai_api_key'


@app.route('/')
def index():
    return "Welcome to the GPT-4 Flask API!"


@app.route('/process', methods=['POST'])
def process_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        text = file.read().decode('utf-8')

        # Call the GPT-4 API using your existing code
        retriever = load_transcript(text)
        mpq_dict = mpq_to_dict()
        rag_chain = init_langchain(retriever)
        response, trio_list = run_query(rag_chain, mpq_dict, retriever)
        print(response)
        # ragas_eval(trio_list)
    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True)
