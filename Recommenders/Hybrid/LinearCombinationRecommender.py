import numpy as np

from Recommenders.BaseRecommender import BaseRecommender


class LinearCombinationRecommender(BaseRecommender):
    def __init__(self, URM_train, recommenders: list[BaseRecommender], weights: list[float]):
        super(LinearCombinationRecommender, self).__init__(URM_train)
        self.recommenders = recommenders
        self.weights = weights
        
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        scores = np.zeros((len(user_id_array), self.n_items))
        for recommender, w in zip(self.recommenders, self.weights):
            scores += recommender._compute_item_score(user_id_array, items_to_compute) * w
        return scores