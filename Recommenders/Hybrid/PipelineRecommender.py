import numpy as np

from Recommenders.BaseRecommender import BaseRecommender


class PipelineRecommender(BaseRecommender):
    def __init__(self, URM_train, recommenders: list[BaseRecommender], cutoffs):
        super(PipelineRecommender, self).__init__(URM_train)
        self.recommenders = recommenders
        self.cutoffs = cutoffs
        
    def recommend(self, user_ids, cutoff=10, remove_seen_flag=True, remove_top_pop_flag=False, return_scores=True, remove_custom_items_flag=True):
        
        recommendations = None
        for recommender_cutoff, recommender in zip(self.cutoffs, self.recommenders):
            recommender_scores = recommender._compute_item_score(user_ids)
            
            for i, user_id in enumerate(user_ids):
                recommender_scores[i, :] = self._remove_seen_on_scores(user_id, recommender_scores[i, :])
            
            if recommendations is None:
                recommendations = np.argsort(recommender_scores, axis=1)[:, -recommender_cutoff:]
                
            else:
                filtered_scores = np.full_like(recommender_scores, -np.inf)
                for i in range(recommendations.shape[0]):
                    row_indices = recommendations[i]
                    filtered_scores[i, row_indices] = recommender_scores[i, row_indices]
                recommendations = np.argsort(filtered_scores, axis=1)[:, -recommender_cutoff:]              
                
        if return_scores:
            return np.flip(recommendations[:, -cutoff:], axis=1), filtered_scores
        return np.flip(recommendations[:, -cutoff:], axis=1)