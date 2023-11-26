import numpy as np


class RoundRobinListCombinationRecommender:
    def __init__(self, URM_train, recommenders):
        # recommenders must be pretrained
        self.URM_train = URM_train
        self.recommenders = recommenders
        
    def recommend(self, user_ids, cutoff=10, remove_seen_flag=True, remove_top_pop_flag=False, return_scores=True, remove_custom_items_flag=True):
        
        final_recommendations = []
        final_scores = []
        for user_id in user_ids:
            
            all_recommendations = []
            all_scores = []
            for recommender in self.recommenders:
                recommendations, scores = recommender.recommend(
                    [user_id],
                    cutoff=cutoff,
                    remove_seen_flag=remove_seen_flag, 
                    remove_top_pop_flag=True,
                    return_scores=True, 
                    remove_custom_items_flag=remove_custom_items_flag
                )
                all_recommendations.append(recommendations[0])
                all_scores.append(scores[0])
               
            recommendations = []
            current_local_index = 0
            while len(recommendations) < cutoff:
                for r in all_recommendations:
                    if len(recommendations) < cutoff and r[current_local_index] not in recommendations:
                        recommendations.append(r[current_local_index])
                current_local_index += 1
                
            final_recommendations.append(recommendations)
            final_scores.append(all_scores[0])
        if return_scores:
            return np.array(final_recommendations), np.array(final_scores)
        return np.array(final_recommendations)
    
    def get_URM_train(self):
        return self.URM_train