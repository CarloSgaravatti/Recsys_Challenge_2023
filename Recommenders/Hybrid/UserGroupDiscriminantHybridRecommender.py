import numpy as np

from Recommenders.BaseRecommender import BaseRecommender


class UserGroupDiscriminantHybridRecommender(BaseRecommender):
    def __init__(self, URM_train, group_id_by_user, recommenders_per_group):
        """
        :param URM_train: user rating matrix.
        :param group_id_by_user: a dictionary that maps a user to the corresponding group id.
        :param recommenders_per_group: a list that contains in element i the recommender to be used for group with id i.
        """
        super(UserGroupDiscriminantHybridRecommender, self).__init__(URM_train)
        self.group_id_by_user = group_id_by_user
        self.recommenders_per_group = recommenders_per_group
        
    def recommend(self, user_ids, cutoff=10, remove_seen_flag=True, remove_top_pop_flag=False, return_scores=True, remove_custom_items_flag=True):
        recommendations = np.empty((0, cutoff), dtype=int)
        scores = np.empty((0, self.n_items), dtype=np.float32)
        for user_id in user_ids:
            user_recommendations, user_scores = self.recommenders_per_group[self.group_id_by_user[user_id]].recommend(
                [user_id],
                cutoff=cutoff,
                remove_seen_flag=remove_seen_flag,
                remove_top_pop_flag=remove_top_pop_flag,
                return_scores=True,
                remove_custom_items_flag=remove_custom_items_flag
            )
            recommendations = np.concatenate([recommendations, user_recommendations], axis=0)
            scores = np.concatenate([scores, user_scores], axis=0)
        
        if return_scores:
            return recommendations, scores
        return recommendations