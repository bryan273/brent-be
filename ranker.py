import numpy as np
import lightgbm as lgb
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
from data_preparator import DataPreparator
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm

class Ranker:
    def __init__(self, num_latent_topics=200):
        self.dictionary = Dictionary()
        self.data_preparator = DataPreparator()
        self.num_latent_topics = num_latent_topics
        self.ranker = None
        self.sentence_model = SentenceTransformer("m-aliabbas1/tiny_bert_29_medicare_intents")
    
    def features(self, query, doc, is_batch=False):
        """
        Fungsi untuk membangun fitur untuk training model
        X: fitur, Y: target
        """

        if not is_batch:
            # Hasil BERT
            v_q = normalize(self.sentence_model.encode(query)).flatten().tolist()
            v_d = normalize(self.sentence_model.encode(doc)).flatten().tolist()
            
            # Fitur tambahan cosine dan jaccard
            cosine_dist = cosine(v_q, v_d)
            jaccard = len(set(query.split()) & set(doc.split())) / len(set(query.split()) | set(doc.split()))
            return v_q + v_d + [jaccard, cosine_dist]
        else:
            # Handle batch queries and documents
            # print(query, doc)
            batch_features = []
            v_q = normalize(self.sentence_model.encode(query)).tolist()
            v_d = normalize(self.sentence_model.encode(doc)).tolist()
                
            for i, (query, doc) in enumerate(zip(query, doc)):

                cosine_dist = cosine(v_q[i], v_d[i])
                jaccard = len(set(query.split()) & set(doc.split())) / len(set(query.split()) | set(doc.split()))
                
                batch_features.append(v_q[i] + v_d[i] + [jaccard, cosine_dist])
            return batch_features

    def prepare_data_train(self, dataset, batch_size=256):
        is_batch = batch_size > 1

        X = []
        Y = []
        batch_queries = []
        batch_docs = []

        for data in tqdm(dataset):
            query, doc, rel = data
            query = ' '.join(query)
            doc = ' '.join(doc)
            
            batch_queries.append(query)
            batch_docs.append(doc)
            Y.append(rel)
            
            if len(batch_queries) == batch_size:
                if is_batch:
                    X.extend(self.features(batch_queries, batch_docs, is_batch))
                else:
                    X.append(self.features(batch_queries, batch_docs, is_batch))
                
                # Clear the batch lists
                batch_queries = []
                batch_docs = []

        # Process any remaining data in the last batch
        if batch_queries:
            if is_batch:
                X.extend(self.features(batch_queries, batch_docs, is_batch))
            else:
                X.append(self.features(batch_queries, batch_docs, is_batch))

        return np.array(X), np.array(Y)
    
    def prepare_data_test(self, dataset, batch_size=256):
        """
        Persiapan data testing (X)
        """
        is_batch = batch_size > 1

        X = []
        batch_queries = []
        batch_docs = []

        for data in dataset:
            query, doc = data
            query = ' '.join(query)
            doc = ' '.join(doc)
            
            batch_queries.append(query)
            batch_docs.append(doc)
            
            if len(batch_queries) == batch_size:
                if is_batch:
                    X.extend(self.features(batch_queries, batch_docs, is_batch))
                else:
                    X.append(self.features(batch_queries, batch_docs, is_batch))
                
                # Clear the batch lists
                batch_queries = []
                batch_docs = []

        # Process any remaining data in the last batch
        if batch_queries:
            if is_batch:
                X.extend(self.features(batch_queries, batch_docs, is_batch))
            else:
                X.append(self.features(batch_queries, batch_docs, is_batch))

        return np.array(X)

    def fit_ranker(self, X_train, Y_train, train_group_qid_count, 
                        X_val=None, Y_val=None, val_group_qid_count=None):
        """
        Fungsi untuk melatih model LGBM Ranker
        """
        
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=100,
            importance_type="gain",
            metric="ndcg",
            num_leaves=40,
            learning_rate=0.02,
            max_depth=-1,
            random_state=42
        )

        if val_group_qid_count != None:
            self.ranker.fit(
                X=X_train, 
                y=Y_train, 
                group=train_group_qid_count,  # Group data for the training set
                eval_at=[5, 10, 25],
                eval_set=[(X_val, Y_val)],  # Evaluation set
                eval_group=[val_group_qid_count],  # Group data for the evaluation set
                eval_names=['val'],  # Name for the evaluation set
                eval_metric=["ndcg@25"],
            )
        else:
            self.ranker.fit(
                X=X_train, 
                y=Y_train, 
                group=train_group_qid_count,  # Group data for the training set
            )
        
    def predict_ranker(self, X):
        """
        Fungsi untuk memprediksi X
        """
        return self.ranker.predict(X)

    def save_model(self, filename, type=''):
        """
        Menyimpan ranker model
        """
        if type=='lgbm':
            self.ranker.booster_.save_model(filename)
        else:
            raise ValueError("Type can only be 'lsi' or 'lgbm'")

    def load_model(self, filename, type=''):
        """
        Load ranker model
        """
        if type=="lgbm":
            self.ranker = lgb.Booster(model_file=filename)
            return self.ranker
        else:
            raise ValueError("Type can only be 'lsi' or 'lgbm'")

if __name__ == "__main__":
    section = ['train_all']

    if 'train_val' in section:
        # Pada 'train_val', data training dan validation dimuat, 
        # model LSI dibuat, dan model ranker LightGBM dilatih. 
        # diperoleh hasil NDCG terbaik pada data validasi.

        train_doc_pkl = r'pickle\train_documents.pkl'
        train_query_pkl = r'pickle\train_queries.pkl'
        val_doc_pkl = r'pickle\val_documents.pkl'
        val_query_pkl = r'pickle\val_queries.pkl'

        # load file dari pickle
        train_documents = DataPreparator.load_from_pickle(train_doc_pkl)
        train_queries = DataPreparator.load_from_pickle(train_query_pkl)
        val_documents = DataPreparator.load_from_pickle(val_doc_pkl)
        val_queries = DataPreparator.load_from_pickle(val_query_pkl)

        # membuat model LSI dari dokumen train
        ranker = Ranker()

        # mempersiapkan train dataset dan validation dataset
        train_qrel_path = r"qrels-folder\train_qrels.txt"
        val_qrel_path = r"qrels-folder\val_qrels.txt"
        train_dataset, train_group_qid_count = ranker.data_preparator.prepare_training_dataset(train_qrel_path, train_queries, train_documents)
        val_dataset, val_group_qid_count = ranker.data_preparator.prepare_training_dataset(val_qrel_path, val_queries, val_documents)
        
        # membuat representasi fitur X dan label Y
        X_train, Y_train = ranker.prepare_data_train(train_dataset)
        X_val, Y_val = ranker.prepare_data_train(val_dataset)

        # melatih model ranker
        ranker.fit_ranker(X_train, Y_train, train_group_qid_count,
                          X_val, Y_val, val_group_qid_count)

        # Setelah melakukan fit pada model
        best_score = ranker.ranker.best_score_['val'] 
        print(f"Best NDCG score on validation set: {best_score}")

        # LSI
        # Output
        # OrderedDict([('ndcg@5', 0.9794901122519907), 
        # ('ndcg@10', 0.9812383590985622), 
        # ('ndcg@25', 0.9814625590191021)])

        # BERT
        # OrderedDict([('ndcg@5', 0.977902698441548), 
        # ('ndcg@10', 0.9806530298234324), 
        # ('ndcg@25', 0.9810282434449141)])
    
    if 'train_all' in section:
        # Pada 'train_all', data training dan validation digabungkan, 
        # model LSI dibuat, dan model ranker LightGBM dilatih. 
        # Model disimpan ke dalam file 'lgbr_base.txt' dan 'lsi_base.model'.

        train_doc_pkl = r'pickle\train_documents.pkl'
        train_query_pkl = r'pickle\train_queries.pkl'
        val_doc_pkl = r'pickle\val_documents.pkl'
        val_query_pkl = r'pickle\val_queries.pkl'

        train_documents = DataPreparator.load_from_pickle(train_doc_pkl)
        train_queries = DataPreparator.load_from_pickle(train_query_pkl)
        val_documents = DataPreparator.load_from_pickle(val_doc_pkl)
        val_queries = DataPreparator.load_from_pickle(val_query_pkl)
        
        documents = {**train_documents, **val_documents}

        ranker = Ranker()

        train_qrel_path = r"qrels-folder\train_qrels.txt"
        val_qrel_path = r"qrels-folder\val_qrels.txt"
        train_dataset, train_group_qid_count = ranker.data_preparator.prepare_training_dataset(train_qrel_path, train_queries, train_documents)
        val_dataset, val_group_qid_count = ranker.data_preparator.prepare_training_dataset(val_qrel_path, val_queries, val_documents)
        
        # data train dan validation digabung kemudian di fit ke X,Y
        X, Y = ranker.prepare_data_train(train_dataset + val_dataset)
        group_qid_count = train_group_qid_count + val_group_qid_count
        ranker.fit_ranker(X, Y, group_qid_count)

        ranker.save_model('lgbr_base.txt', type='lgbm')

    if 'trial' in section:
        # Pada 'trial', contoh penggunaan model yang sudah ada. 
        # Dokumen dan query dimuat, model LSI dimuat, dan 
        # diprint beberapa prediksi peringkat.

        doc_pkl = r'pickle\train_documents.pkl'
        query_pkl = r'pickle\train_queries.pkl'

        documents = DataPreparator.load_from_pickle(doc_pkl)
        queries = DataPreparator.load_from_pickle(query_pkl)

        ranker = Ranker()

        train_qrel_path = r"qrels-folder\train_qrels.txt"
        dataset, group_qid_count = ranker.data_preparator.prepare_training_dataset(train_qrel_path, queries, documents)
        
        X, Y = ranker.prepare_data_train(dataset)

        ranker.load_model('lgbr_base.txt', type='lgbm')
        print(ranker.predict_ranker(X[:2]))