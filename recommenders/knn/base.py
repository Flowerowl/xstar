from ..base import MemoryBasedRecommender


class ItemRecommender(MemoryBasedRecommender):
    def most_similar_items(self, item_id, how_many=None):
        raise NotImplementError('ItemRecommender is an abstract class.')

    def recommend_because(self, user_id, item_id, how_many, **params):
        raise NotImplementError('ItemRecommender is an abstract class.')


class UserRecommender(MemoryBasedRecommender):
    def most_similar_items(self, user_id, how_many=None):
        raise NotImplementError('UserRecommender is an abstract class.')

    def recommend_because(self, user_id, item_id, how_many, **params):
        raise NotImplementError('UserRecommender is an abstract class.')


class BaseCandidateItemsStrategy(object):
    def candidate_items(self, user_id, data_model, **params):
        raise NotImplementError('BaseCandidateItemsStrategy is an abstract class.')


class BaseUserNeighborhoodStrategy(object):
    def user_neighborhood(self, user_id, data_model, n_similiarity='user_similiarity',
            distance=None, n_users=None, **params):
        raise NotImplementError('BaseCandidateItemsStrategy is an abstract class.')
