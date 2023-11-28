from retrieval import Retrieval
from data_preparator import DataPreparator
from letor import LETOR
import re

def main_pipeline_letor(query, k=50):

    clean_text = " ".join(re.findall(r"\w+", query))
    terms = retrieval.data_preparator.preprocess_text(clean_text).split(' ')

    doc_paths, _ = retrieval.retrieve_documents(terms)
    doc_ids = [doc_path.split('\\')[-1][:-4] for doc_path in doc_paths]
    dataset = retrieval.data_preparator.prepare_testing_dataset(terms, doc_paths)

    # melakukan preparation data test dan prediksi 
    X = letor.ranker.prepare_data_test(dataset)
    pred = letor.ranker.predict_ranker(X) +  X[:,-1]/100
    doc_pred_pairs = list(zip(doc_ids, pred))
    sorted_pairs = sorted(doc_pred_pairs, key=lambda x: x[1], reverse=True)
    top_k_doc_ids_letor = [pair[0] for pair in sorted_pairs]
    top_k_doc_ids_bsbi = doc_ids

    intersection = [x for x in top_k_doc_ids_letor if x in top_k_doc_ids_bsbi]
    intersection = intersection + [x for x in top_k_doc_ids_letor if x not in intersection]

    return intersection[:k]

if __name__=="__main__":
    ranker_path = "lgbr_base.txt"
    letor = LETOR(ranker_path)
    retrieval = Retrieval()

    mapping_file = DataPreparator.load_from_pickle("pickle/mapping_doc.pkl")
    query = "cardiovascular disease"
    docs = main_pipeline_letor(query, k=50)
    for doc in docs:
        with open("main_pipeline.txt", "w") as output_file:
            output_file.write("Query : " + query + "\n")
            for doc in docs:
                with open(mapping_file[doc], "r") as file:
                    content = file.read()
                    output_file.write(doc + " : " + content + "\n")