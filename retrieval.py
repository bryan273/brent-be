from bsbi import BSBIIndex
from compression import VBEPostings
from data_preparator import DataPreparator
from tqdm import tqdm

class Retrieval:
    def __init__(self, k=100):
        self.BSBI_instance = BSBIIndex(data_dir='collections',
                                       postings_encoding=VBEPostings,
                                       output_dir='index')
        self.data_preparator = DataPreparator()
        self.k = k

    def retrieve_documents(self, query):
        """
        Mengambil 100 dokumen teratas dan skornya menggunakan indeks BSBI.
        """

        results = self.BSBI_instance.retrieve_tfidf(' '.join(query), k=self.k)
        doc_paths = [res[1] for res in results]
        scores = [res[0] for res in results]

        return doc_paths, scores
    
    def process_testing_query(self, qid_query):
        """
        Fungsi untuk memproses query testing.

        Format result: {qid: (dataset, doc id, score bsbi)}
        """
        qid_data = {}
        for key in tqdm(qid_query):
            query = qid_query[key]
            doc_paths, raw_scores = self.retrieve_documents(query)

            doc_ids = [doc_path.split('\\')[-1][:-4] for doc_path in doc_paths]
            dataset = self.data_preparator.prepare_testing_dataset(query, doc_paths)
        
            qid_data[key] = (dataset, doc_ids, raw_scores)

        return qid_data