from data_preparator import DataPreparator
from ranker import Ranker
import pandas as pd
import math
from tqdm import tqdm

class LETOR:
    def __init__(self, ranker_path, data_path=None):
        self.ranker = Ranker()
        self.ranker.load_model(ranker_path, type="lgbm")
    
        self.data_preparator = DataPreparator()
        if data_path:
            self.data = DataPreparator.load_from_pickle(data_path)

    def ndcg(ranking):
        """ Calculate the normalized search effectiveness metric score 
            with Normalized Discounted Cumulative Gain.

            Parameters
            ----------
            ranking: List[int]
                Binary vector such as [1, 0, 1, 1, 1, 0]
                Relevance of documents in ranking order.
                Example: [1, 0, 1, 1, 1, 0]

            Returns
            -------
            Float
                nDCG score
        """

        def dcg(ranking):
            # Existing DCG calculation
            score = 0.0
            for i in range(len(ranking)):
                rel_i = ranking[i]
                if rel_i == 1:
                    score += 1.0 / (math.log2(i + 2))  # i+2 because index starts from 0
            return score
        
        # Calculate the DCG of the actual ranking
        actual_dcg = dcg(ranking)

        # Sort the ranking to get the ideal ranking and calculate the IDCG
        ideal_ranking = sorted(ranking, reverse=True)
        ideal_dcg = dcg(ideal_ranking)

        # Handle the case where IDCG is zero to avoid division by zero
        if ideal_dcg == 0:
            return 0

        # Calculate the nDCG
        return actual_dcg / ideal_dcg

if __name__ == "__main__":

    section = ['testing']

    if 'testing' in section:
        # memuat model ranker, testing data, dan file qrels (yang berisi <qid,doc> yang relevan).
        ranker_path = "lgbr_base.txt"
        data_path = r'pickle\test_data.pkl'
        qrel_path = r"qrels-folder\test_qrels.txt"

        letor = LETOR(ranker_path, data_path)
        qrels = letor.data_preparator.read_qrels_test(qrel_path)

        data = letor.data

        ndcgs_letor = []
        ndcgs_bsbi = []

        columns = ['query_id', 'doc_id', 'score', 'relevance']
        df_letor = pd.DataFrame(columns=columns)
        df_bsbi = pd.DataFrame(columns=columns)

        # iteratre untuk setiap query
        for qid in tqdm(qrels):
            dataset, doc_ids, raw_scores = data[qid]
            qrel = qrels[qid]
            k = 25

            # melakukan preparation data test dan prediksi 
            X = letor.ranker.prepare_data_test(dataset)
            pred = letor.ranker.predict_ranker(X)
            
            # untuk mendapatkan top k dokumen
            doc_pred_pairs = list(zip(doc_ids, pred))
            sorted_pairs = sorted(doc_pred_pairs, key=lambda x: x[1], reverse=True)
            top_k_doc_ids_letor = [pair[0] for pair in sorted_pairs[:k]]
            top_k_doc_ids_letor = ['D'+ doc for doc in top_k_doc_ids_letor]

            top_k_doc_ids_bsbi = doc_ids[:k]
            top_k_doc_ids_bsbi = ['D'+ doc for doc in top_k_doc_ids_bsbi]

            # mendapatkan ranking dari top k doc yang di sort
            ranking_letor = [qrel[did] for did in top_k_doc_ids_letor]
            ranking_bsbi = [qrel[did] for did in top_k_doc_ids_bsbi]
            
            # Hasil nDCG untuk model "letor" dan model "bsbi" akan dihitung dan disimpan.
            ndcg_score_letor = LETOR.ndcg(ranking_letor)
            ndcg_score_bsbi = LETOR.ndcg(ranking_bsbi)

            ndcgs_letor.append(ndcg_score_letor)
            ndcgs_bsbi.append(ndcg_score_bsbi)
            
            # Create DataFrames for 'letor' and 'bsbi' scores in the current loop
            df_letor_loop = pd.DataFrame({'query_id': [qid]*len(ranking_letor),
                                        'doc_id': top_k_doc_ids_letor,
                                        'score': [pair[1] for pair in sorted_pairs[:k]],
                                        'relevance': ranking_letor})
            
            df_bsbi_loop = pd.DataFrame({'query_id': [qid]*len(ranking_letor),
                                        'doc_id': top_k_doc_ids_bsbi,
                                        'score': raw_scores[:k],
                                        'relevance': ranking_bsbi})

            # Concatenate the DataFrames to the main DataFrames
            df_letor = pd.concat([df_letor, df_letor_loop], ignore_index=True)
            df_bsbi = pd.concat([df_bsbi, df_bsbi_loop], ignore_index=True)

        print("Mean NDCG Score for Letor:", sum(ndcgs_letor) / len(ndcgs_letor))
        print("Mean NDCG Score for BSBI:", sum(ndcgs_bsbi) / len(ndcgs_bsbi))
        
        # Save df_letor to a CSV file
        df_letor.to_csv('df_letor.csv', index=False)

        # Save df_bsbi to a CSV file
        df_bsbi.to_csv('df_bsbi.csv', index=False)