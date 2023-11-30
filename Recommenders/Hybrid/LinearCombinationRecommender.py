import numpy as np
import scipy.sparse as sps

from Recommenders.BaseRecommender import BaseRecommender


class LinearCombinationRecommender(BaseRecommender):
    def __init__(self, URM_train, recommenders: list[BaseRecommender], weights: list[float]):
        super(LinearCombinationRecommender, self).__init__(URM_train)
        self.recommenders = recommenders
        self.weights = weights

    def _calculate_top_pop_items(self, cutoff):
        item_popularity = np.ediff1d(sps.csc_matrix(self.URM_train).indptr)
        self.filterTopPop = True
        self.filterTopPop_ItemsID = item_popularity.argsort()[-cutoff:]
        
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        scores = np.zeros((len(user_id_array), self.n_items))
        for recommender, w in zip(self.recommenders, self.weights):
            scores += recommender._compute_item_score(user_id_array, items_to_compute) * w
        return scores