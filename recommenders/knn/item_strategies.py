#encoding:utf-8
from __future__ import unicode_literals

from base import BaseCandidateItemsStrategy
import numpy as np


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
