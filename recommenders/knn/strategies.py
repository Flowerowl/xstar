#encoding:utf-8
from __future__ import unicode_literals

import numpy as np

from .base import BaseCandidateItemsStrategy, BaseUserNeighborhoodStrategy
from ...similarities.similarities import UserSimilarity
from ...metrics.pairwise import euclidean_distances


class AllPossibleItemsStrategy(BaseCandidateItemsStrategy):
    def candidate_items(self, user_id, data_model, **params):
        preferences = data_model.items_from_user(user_id)
        possible_items = data_model.item_ids()
        # 返回除去用户评分过的其他item ids
        return np.setdiff1d(possible_items, preferences, assume_unique=True)


class ItemsNeighborhoodStrategy(BaseCandidateItemsStrategy):
    def candidate_items(self, user_id, data_model, **params):
        #用户评分过得item ids
        preferences = data_model.items_from_user(user_id)
        possible_items = np.array([])
        for item_id in preferences:
            #该item id 对应的所有用户id 以及评分[('Leopoldo Pires', 3.0),..]
            item_preferences = data_model.preferences_for_item(item_id)
            if data_model.has_preference_values():
                for user_id, score in item_preferences:
                    #遍历对item_id评分过的用户的其他评分物品
                    possible_items = np.append(possible_items, \
                        data_model.items_from_user(user_id))
            else:
                for user_id in item_preferences:
                    possible_items = np.append(possible_items, \
                        data_model.items_from_user(user_id))
        #对item id去重
        possible_items = np.unique(possible_items)
        # 返回除去用户评分过的其他item ids
        return np.setdiff1d(possible_items, preferences, assume_unique=True)


class AllNeighborsStrategy(BaseUserNeighborhoodStrategy):
    def user_neighborhood(self, user_id, data_model, similarity='user_similarity',
        distance=None, nhood_size=None, **params):
        user_ids = data_model.user_ids()
        return user_ids[user_ids != user_id] if user_ids.size else user_ids


class NearestNeighborsStrategy(BaseUserNeighborhoodStrategy):
    def __init__(self):
        self.similarity = None

    def _sampling(self, data_model, sampling_rate):
        return data_model

    def _set_similarity(self, data_model, similarity, distance, nhood_size):
        if not isinstance(self.similarity, UserSimilarity) \
             or not distance == self.similarity.distance:
            nhood_size = nhood_size if not nhood_size else nhood_size + 1
            self.similarity = UserSimilarity(data_model, distance, nhood_size)

    def user_neighborhood(self, user_id, data_model, n_similarity='user_similarity',
             distance=None, nhood_size=None, **params):
        minimal_similarity = params.get('minimal_similarity', 0.0)
        sampling_rate = params.get('sampling_rate', 1.0)

        data_model = self._sampling(data_model, sampling_rate)
        if distance is None:
            distance = euclidean_distances
        if n_similarity == 'user_similarity':
            self._set_similarity(data_model, n_similarity, distance, nhood_size)
        else:
            raise ValueError('similarity argument must be user_similarity')
        #[('Marcel Caraciolo', 0.99124070716193036), ('Steve Gates', 0.92447345164190498), ('Lorena Abreu', 0.89340514744156441), ('Sheldom', 0.66284898035987017), ('Paola Pow', 0.3812464258315118), ('Leopoldo Pires', -0.99999999999999978)]
        #import pdb;pdb.set_trace()
        neighborhood = [to_user_id for to_user_id, score in self.similarity[user_id] \
                           if not np.isnan(score) and score >= minimal_similarity and \
                           user_id != to_user_id]
        return neighborhood
