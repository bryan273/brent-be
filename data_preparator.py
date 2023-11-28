import re
import random
import pickle
import os
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
nltk.data.path.append("./nltk_data")
from nltk.stem import PorterStemmer

class DataPreparator:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """
        Melakukan preprocessing teks, termasuk stemming dan penghapusan stop words.
        """
        words = nltk.word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        filtered_words = [word for word in stemmed_words if word.lower() not in self.stop_words]
        preprocessed_text = ' '.join(filtered_words)
        return preprocessed_text

    def process_file(self, file_path):
        """
        Menghasilkan map yang memetakan ID ke terms konten setelah preprocessing.

        Result format: {id: term1, id: term2, ..}
        """
        result = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            total_lines = sum(1 for _ in file)
            file.seek(0)
            for line in tqdm(file, total = total_lines):
                splitted = line.split()
                key = splitted[0]
                text = ' '.join(splitted[1:])

                clean_text = " ".join(re.findall(r"\w+", text))
                terms = self.preprocess_text(clean_text).split(' ')
                result[key] = terms

        return result
    
    def create_map_doc(self, collections_dir):
        """
        Membuat mapping dokumen yang mengaitkan ID dokumen dengan 
        path file dokumen yang sesuai.

        Result format: {did1: path_did1, did2: path_did2, ..}
        """

        did_file = {}

        # subfolders (0-8)
        for folder_name in os.listdir(collections_dir):
            folder_path = os.path.join(collections_dir, folder_name)
                
            # text files in the subfolder
            for file_name in (os.listdir(folder_path)):
                if file_name.endswith(".txt"):
                    file_id = os.path.splitext(file_name)[0]
                    file_path = os.path.join(folder_path, file_name)
                    
                    did_file[file_id] = file_path
                    
        return did_file

    def process_validation_documents(self, dids, did_file):
        """
        Memproses doc validation yang merupakan mapping dari ID dokumen
        dan terms yang diberikan dan melakukan preprocessing pada termsnya.

        Result format: {did1: term1, did2: term2, ..}
        """
        result = {}

        for did in tqdm(dids):
            file_path = did_file[did]
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                clean_text = " ".join(re.findall(r"\w+", content))
                terms = self.preprocess_text(clean_text).split(' ')
                result[did] = terms

        return result

    def save_to_pickle(data, file_name):
        """
        Menyimpan objek ke dalam file pickle jika file tersebut belum ada
        """
        if not os.path.exists(file_name):
            with open(file_name, 'wb') as file:
                pickle.dump(data, file)
            print(f'Object saved as {file_name}')
        else:
            print(f'File already exists at {file_name}, not saving again.')

    def load_from_pickle(file_name):
        """
        Load objek jika file tersebut ada
        """
        if os.path.exists(file_name):
            print(f'Object loaded from {file_name}')
            with open(file_name, 'rb') as file:
                return pickle.load(file)
        else:
            print(f'File not exists at {file_name}')

    def read_qrels(self, file_path):
        """
        Fungsi untuk membaca qrels khususnya untuk qrel train dan validation

        Relevance level: 3 (fully relevant), 2 (partially relevant), 1 (marginally relevant)
        Format result: {qid1 : {did1: 3, did2: 3, ...}, qid2: ....}
        """
        qrels = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    query_id, doc_id, relevance = parts
                    relevance = int(relevance)
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = relevance

        return qrels

    def read_qrels_test(self, qrel_file):
        """
        Fungsi untuk membaca qrels untuk qrel test
        1 bila exist dan 0 bila tidak exist
        Format result: {qid1 : {did1: 1, did2: 0, ...}, qid2: ....}
        """
        qrels = defaultdict(lambda: defaultdict(lambda: 0)) 
        with open(qrel_file) as file:
            for line in file:
                parts = line.strip().split()
                qid = parts[0]
                did = parts[1]
                qrels[qid][did] = 1

        return qrels

    def prepare_training_dataset(self, qrels_path, queries, documents, num_negatives=1):
        """
        Mempersiapkan dataset pelatihan dengan menggunakan file qrels sebagai referensi.

        Result format dataset: list of (query, document, relevansi)
        Result format group qid count: list of group
        """
        qrels = self.read_qrels(qrels_path)
        q_docs_rel = {}

        # Membuat dict yang berisi query-document relevansi
        for q_id in qrels:
            if q_id in queries:
                relevant_docs = [(doc_id, rel) for doc_id, rel in qrels[q_id].items() if doc_id in documents]
                q_docs_rel[q_id] = relevant_docs

        # Menghitung jumlah data dalam setiap grup query
        group_qid_count = []
        dataset = []

        # Membangun dataset pelatihan dengan relevansi 1,2,3
        for q_id, docs_rels in q_docs_rel.items():
            group_qid_count.append(len(docs_rels) + num_negatives)
            dataset.extend([(queries[q_id], documents[doc_id], rel) for doc_id, rel in docs_rels])
            dataset.extend([(queries[q_id], random.choice(list(documents.values())), 0) for _ in range(num_negatives)])

        return dataset, group_qid_count
    
    def prepare_testing_dataset(self, query, doc_paths):
        """
        Processes the retrieved documents by reading and pre-processing their content.
        """
        dataset = []
        for file_path in doc_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                tokens = re.findall(r"\w+", content)
                text_clean = " ".join(tokens)
                terms = self.preprocess_text(text_clean).split(' ')

            dataset.append((query, terms))
        return dataset
    
if __name__ == "__main__":

    # Bagian ini adalah bagian utama program yang akan dieksekusi saat script dijalankan.

    data_preparator = DataPreparator()
    section = ['train', 'validation', 'testing']

    if 'train' in section:
        # Bagian ini digunakan untuk memproses data pelatihan.
        # Misalnya, membaca query dan dokumen, kemudian menyimpannya dalam format pickle.

        did_doc = data_preparator.process_file('qrels-folder/train_docs.txt')
        qid_query = data_preparator.process_file('qrels-folder/train_queries.txt')

        doc_pkl = r'pickle\train_documents.pkl'
        query_pkl = r'pickle\train_queries.pkl'

        DataPreparator.save_to_pickle(did_doc, doc_pkl)
        DataPreparator.save_to_pickle(qid_query, query_pkl)

    if 'validation' in section:
        # Bagian ini digunakan untuk memproses data validasi.
        # Misalnya, membaca query validasi dan menyimpannya dalam format pickle.

        did_doc = data_preparator.process_file('qrels-folder/val_docs.txt')
        qid_query = data_preparator.process_file('qrels-folder/val_queries.txt')

        doc_pkl = r'pickle\val_documents.pkl'
        query_pkl = r'pickle\val_queries.pkl'
        
        DataPreparator.save_to_pickle(did_doc, doc_pkl)
        DataPreparator.save_to_pickle(qid_query, query_pkl)

    if 'testing' in section:
        # Bagian ini digunakan untuk memproses data pengujian.
        # Misalnya, membaca query pengujian dan menyimpannya dalam format pickle.
        qid_query = data_preparator.process_file('qrels-folder/test_queries.txt')

        query_pkl = r'pickle\test_queries.pkl'
        DataPreparator.save_to_pickle(qid_query, query_pkl)