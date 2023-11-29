import numpy as np
import lightgbm as lgb
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
from data_preparator import DataPreparator
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm
import joblib

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

    def save_model(self, filename):
        """
        Menyimpan ranker model
        """
        ext = filename.split(".")[-1]
        if (ext=="txt") or (ext=='mdl'):
            self.ranker.booster_.save_model(filename)
        elif ext=='pkl':
            joblib.dump(self.ranker, filename)
        else:
            raise ValueError("Extension can only be 'txt' or 'pkl'")

    def load_model(self, filename):
        """
        Load ranker model
        """
        ext = filename.split(".")[-1]
        if (ext=="txt") or (ext=='mdl'):
            self.ranker = lgb.Booster(model_file=filename)
            return self.ranker
        elif ext=='pkl':
            return joblib.load(filename)
        else:
            raise ValueError("Extension can only be 'txt' or 'pkl'")