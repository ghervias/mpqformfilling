from flask import Flask, request, jsonify
import openai
from werkzeug.utils import secure_filename

from transcript_loader import load_transcript
from gpt_setup import mpq_to_dict, init_langchain
from run_query import run_query
from run_ragas import ragas_eval
import langchain  # Make sure you import the necessary parts of langchain

ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.json.sort_keys = False


# Configure your OpenAI API key
openai.api_key = 'your_openai_api_key'


@app.route('/')
def index():
    return "Welcome to the GPT-4 Flask API!"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process', methods=['POST'])
def process_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        text = file.read().decode('utf-8')

        # Call the GPT-4 API using your existing code
        retriever = load_transcript(text)
        mpq_dict = mpq_to_dict()
        rag_chain = init_langchain(retriever)
        response, trio_list = run_query(rag_chain, mpq_dict, retriever)
        print(response)
        print(jsonify(response))
        return jsonify(response)
        # ragas_eval(trio_list)
    return "Invalid file type."



if __name__ == '__main__':
    app.run(debug=True)
