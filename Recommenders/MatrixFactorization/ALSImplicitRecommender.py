import implicit
import numpy as np
import os

from Recommenders.BaseRecommender import BaseRecommender


class ALSImplicitRecommender(BaseRecommender):
    def __init__(self, URM_train):
        super(ALSImplicitRecommender, self).__init__(URM_train)
        
    def fit(self, factors=50, iterations=10, regularization=1e-2, alpha=1, random_state=None):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            alpha=alpha,
            dtype=np.float32,
            random_state=random_state
        )
        self.model.fit(self.URM_train)
        
    def recommend(self, user_id_array, cutoff=10, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        
        ids, scores = self.model.recommend(user_id_array, self.URM_train[user_id_array], N=cutoff, 
                                           items=items_to_compute, filter_already_liked_items=remove_seen_flag)
        ids = np.array(ids)
        scores = np.array(scores)
        if return_scores:
            total_scores = np.ones((len(user_id_array), self.n_items)) * (-np.inf)
            for i in range(len(user_id_array)):
                total_scores[i, ids[i, :]] = scores[i, :]
            return ids, total_scores
        return ids
    
    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = 'als'
        self.model.save(os.path.join(folder_path, file_name))
        
    def load_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = 'als.npz'
        self.model = implicit.cpu.als.AlternatingLeastSquares.load(os.path.join(folder_path, file_name))