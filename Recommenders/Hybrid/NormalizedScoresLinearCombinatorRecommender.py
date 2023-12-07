import numpy as np
import scipy.sparse as sps
from numpy import linalg as LA

from Recommenders.BaseRecommender import BaseRecommender


class NormalizedScoresLinearCombinationRecommender(BaseRecommender):
    def __init__(self, URM_train, recommenders: list[BaseRecommender], weights: list[float], normalize=None):
        super(NormalizedScoresLinearCombinationRecommender, self).__init__(URM_train)
        self.recommenders = recommenders
        weights_sum = sum(weights)
        self.weights = [w / weights_sum for w in weights]
        if normalize is not None and normalize not in ['l1', 'l2', 'linf', 'lminusinf']:
            raise ValueError('Normalization term not recognized')
        self.normalize = normalize
        
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        scores = np.zeros((len(user_id_array), self.n_items))
        for recommender, w in zip(self.recommenders, self.weights):
            recommender_scores = recommender._compute_item_score(user_id_array, items_to_compute)
            scores += self._normalize_scores(recommender_scores) * w
        return scores
    
    def _normalize_scores(self, scores):       
        if self.normalize is None:
            return scores
        elif self.normalize == 'l1':
            return scores / (LA.norm(scores, 1, axis=1, keepdims=True) + 1e-6)
        elif self.normalize == 'l2':
            return scores / (LA.norm(scores, 2, axis=1, keepdims=True) + 1e-6)
        elif self.normalize == 'linf':
            return scores / (LA.norm(scores, np.inf, axis=1, keepdims=True) + 1e-6)
        else:
            return scores / (LA.norm(scores, -np.inf, axis=1, keepdims=True) + 1e-6)  