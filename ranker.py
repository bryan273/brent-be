import numpy as np
import lightgbm as lgb
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
from data_preparator import DataPreparator

class Ranker:
    def __init__(self, num_latent_topics=200):
        self.dictionary = Dictionary()
        self.data_preparator = DataPreparator()
        self.num_latent_topics = num_latent_topics
        self.model = None
        self.ranker = None
    
    def create_lsi_model(self, documents):
        """
        Fungsi untuk membuat model LSI
        """
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in documents.values()]
        self.model = LsiModel(bow_corpus, num_topics=self.num_latent_topics, random_seed=42)

    def vector_representation(self, text):
        """
        Fungsi untuk mengubah representasi teks menjadi vektor dengan LSI
        """
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == self.num_latent_topics else [0.] * self.num_latent_topics

    def features(self, query, doc):
        """
        Fungsi untuk membangun fitur untuk training model
        X: fitur, Y: target
        """

        # Hasil LSI
        v_q = self.vector_representation(query)
        v_d = self.vector_representation(doc)
        
        # Fitur tambahan cosine dan jaccard
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(set(query) & set(doc)) / len(set(query) | set(doc))
        return v_q + v_d + [jaccard, cosine_dist]

    def prepare_data_train(self, dataset):
        """
        Persiapan data training (X,Y)
        """
        X = []
        Y = []
        for data in dataset:
            query, doc, rel = data
            X.append(self.features(query, doc))
            Y.append(rel)
        return np.array(X), np.array(Y)

    def prepare_data_test(self, dataset):
        """
        Persiapan data testing (X)
        """
        X = []
        for (query, doc) in dataset:
            X.append(self.features(query, doc))
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
        if type=='lsi':
            self.model.save(filename)
        elif type=='lgbm':
            self.ranker.booster_.save_model(filename)
        else:
            raise ValueError("Type can only be 'lsi' or 'lgbm'")

    def load_model(self, filename, type=''):
        """
        Load ranker model
        """
        if type=='lsi':
            self.model = LsiModel.load(filename)
            return self.model
        elif type=="lgbm":
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
        ranker.create_lsi_model(train_documents)

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

        # Output
        # OrderedDict([('ndcg@5', 0.9794901122519907), 
        # ('ndcg@10', 0.9812383590985622), 
        # ('ndcg@25', 0.9814625590191021)])
    
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
        ranker.create_lsi_model(documents)

        train_qrel_path = r"qrels-folder\train_qrels.txt"
        val_qrel_path = r"qrels-folder\val_qrels.txt"
        train_dataset, train_group_qid_count = ranker.data_preparator.prepare_training_dataset(train_qrel_path, train_queries, train_documents)
        val_dataset, val_group_qid_count = ranker.data_preparator.prepare_training_dataset(val_qrel_path, val_queries, val_documents)
        
        # data train dan validation digabung kemudian di fit ke X,Y
        X, Y = ranker.prepare_data_train(train_dataset + val_dataset)
        group_qid_count = train_group_qid_count + val_group_qid_count
        ranker.fit_ranker(X, Y, group_qid_count)

        ranker.save_model('lgbr_base.txt', type='lgbm')
        ranker.save_model('lsi_base.model', type='lsi')

    if 'trial' in section:
        # Pada 'trial', contoh penggunaan model yang sudah ada. 
        # Dokumen dan query dimuat, model LSI dimuat, dan 
        # diprint beberapa prediksi peringkat.

        doc_pkl = r'pickle\train_documents.pkl'
        query_pkl = r'pickle\train_queries.pkl'

        documents = DataPreparator.load_from_pickle(doc_pkl)
        queries = DataPreparator.load_from_pickle(query_pkl)

        ranker = Ranker()
        ranker.create_lsi_model(documents)

        train_qrel_path = r"qrels-folder\train_qrels.txt"
        dataset, group_qid_count = ranker.data_preparator.prepare_training_dataset(train_qrel_path, queries, documents)
        
        X, Y = ranker.prepare_data_train(dataset)

        ranker.load_model('lgbr_base.txt', type='lgbm')
        print(ranker.predict_ranker(X[:2]))