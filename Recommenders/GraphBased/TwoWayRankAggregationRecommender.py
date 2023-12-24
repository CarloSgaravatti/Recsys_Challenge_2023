import numpy as np

from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.HHPRecommender import HHPRecommender
from Recommenders.BaseRecommender import BaseRecommender


class TwoWayRankAggregationRecommender(BaseRecommender):
    '''
    TWRA (Two Way Rank Aggregation) is an algorithm designed to work with graph algorithms that computes
    score of user u and item i as a convex combination of type:
    
    score(u, i) = (1 - lambda) * f-score(u, i) + lambda * b-score(u, i)
    
    where f-score is obtained by apply a graph based algorithm (ProbS, HeatS, P3alpha, RP3beta, HHP, ...)
    in the usual way and b-score is obtained by applying the same (or another) algorithm but with the role 
    of users and items reversed. For example, with ProbS (P3), the 3 step random walk starts from an item 
    and not from a user.

    In principle, this idea can be extended also to algorithm that do not work with graphs.
    
    Original Paper: https://arxiv.org/pdf/2004.10393.pdf
    '''
    
    RECOMMENDER_NAME = "TwoWayRankAggregationRecommender"
    AVAILABLE_ALGORITHMS = ['p3', 'p3alpha', 'rp3beta', 'hhp']
    
    def __init__(self, URM_train):
        super(TwoWayRankAggregationRecommender, self).__init__(URM_train)
        
        self.bURM_train = self.URM_train.transpose(copy=True).tocsr()
        
    def fit(
        self, 
        base_f_algorithm: str = 'p3', 
        base_b_algorithm: str = 'p3', 
        convex_lambda: float = 0.5, # must be between 0 and 1
        f_params: dict = None, 
        b_params: dict = None
    ):
        
        if base_f_algorithm not in self.AVAILABLE_ALGORITHMS:
            raise ValueError("Value for 'base_f_algorithm' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_ALGORITHMS, base_f_algorithm))
            
        if base_b_algorithm not in self.AVAILABLE_ALGORITHMS:
            raise ValueError("Value for 'base_b_algorithm' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_ALGORITHMS, base_b_algorithm))
            
        if convex_lambda < 0.0 or convex_lambda > 1.0:
            raise ValueError("Value for 'convex_lambda' not valid. Must be inside [0, 1]")
        
        self.base_f_algorithm = base_f_algorithm
        self.base_b_algorithm = base_b_algorithm
        self.convex_lambda = convex_lambda
        self.f_params = f_params if f_params is not None else {}
        self.b_params = b_params if b_params is not None else {}

        if self.base_f_algorithm == 'p3':
            if 'alpha' in self.f_params:
                self.f_params['alpha'] = 1.0
            self.f_recommender = P3alphaRecommender(self.URM_train)
        elif self.base_f_algorithm == 'p3alpha':
            self.f_recommender = P3alphaRecommender(self.URM_train)
        elif self.base_f_algorithm == 'rp3beta':
            self.f_recommender = RP3betaRecommender(self.URM_train)
        else:
            self.f_recommender = HHPRecommender(self.URM_train)
            
        if self.base_b_algorithm == 'p3':
            if 'alpha' in self.b_params:
                self.b_params['alpha'] = 1.0
            self.b_recommender = P3alphaRecommender(self.bURM_train)
        elif self.base_b_algorithm == 'p3alpha':
            self.b_recommender = P3alphaRecommender(self.bURM_train)
        elif self.base_b_algorithm == 'rp3beta':
            self.b_recommender = RP3betaRecommender(self.bURM_train)
        else:
            self.b_recommender = HHPRecommender(self.bURM_train)
            
        self.f_recommender.fit(**self.f_params)
        self.b_recommender.fit(**self.b_params)
        
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        
        f_scores = self.f_recommender._compute_item_score(user_id_array, items_to_compute)
        
        if items_to_compute:
            b_scores = -np.inf * np.ones((self.n_items, len(user_id_array)))
            b_scores[items_to_compute, :] = self.b_recommender._compute_item_score(items_to_compute, items_to_compute=user_id_array)[:, user_id_array]
        else:
            b_scores = self.b_recommender._compute_item_score(np.arange(self.n_items), items_to_compute=user_id_array)[:, user_id_array]
        
        return (1 - self.convex_lambda) * f_scores + self.convex_lambda * b_scores.T