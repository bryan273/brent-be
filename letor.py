from data_preparator import DataPreparator
from ranker import Ranker
import pandas as pd
import math

class LETOR:
    def __init__(self, ranker_path, data_path=None):
        self.ranker = Ranker()
        self.ranker.load_model(ranker_path)
    
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