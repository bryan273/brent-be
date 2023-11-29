from flask import Flask, request, jsonify
from data_preparator import DataPreparator
import json
from letor import LETOR
from flask_cors import CORS

from main_pipeline import get_relevant_doc_id
from retrieval import Retrieval

app = Flask(__name__)
CORS(app)

mapping_file = DataPreparator.load_from_pickle("pickle/mapping_doc.pkl")
ranker_path = "lgbr_base.txt"
letor = LETOR(ranker_path)
retrieval = Retrieval()

@app.route("/")
def hello_world():
    return "Hello"

@app.route("/doc/<string:doc_id>")
def get_document_by_id(doc_id):
    path = mapping_file[doc_id]
    with open(path, "rb") as file:
        content = file.read()

    return json.dumps({
        "document_id": doc_id,
        "content": content.decode('utf-8')
    })

@app.route("/search", methods=['GET'])
def get_rel_docs():
    query = request.args.to_dict()['q']
    doc_paths = get_relevant_doc_id(letor, retrieval, query)
    response = []
    for doc_id in doc_paths:
        path = mapping_file[doc_id]
        with open(path, "rb") as file:
            content = file.read()
        content = content.decode('utf-8')
        response.append({"doc_id": doc_id, "content": content})
    return jsonify(response)
