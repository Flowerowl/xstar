#encoding:utf-8
from __future__ import unicode_literals

import numpy as np
from base import BaseSimiliarity
from ..metrics.pairwise import loglikehood_coefficient


def find_common_elements(source, target):
    source = dict(source)
    target = dict(target)
    inter = np.intersect1d(source.keys(), target.keys())
    common_preferences = zip(*[(source[item], target[item]) for item in \
            inter if not np.isnan(source[item]) and not np.isnan(target[item])])
    if common_preferences:
        return np.asarray([common_preferences[0]]), np.asarray(common_preferences[1])
    else:
        return np.asarray([[]]), np.asarray([[]])


class UserSimilarity(BaseSimiliarity):
    def __init__(self, model, distance, number=None):
        BaseSimiliarity.__init__(self, model, distance, number)

    def get_similarity(self, source_id, target_id):
        source_preferences = self.model.preferences_from_user(source_id)
        target_preferences = self.model.preferences_from_user(target_id)

        if self.model.has_preference_values():
            source_preferences, target_preferences = \
                find_common_elements(source_preferences, target_preferences)

        if source_preferences.ndim == 1 and target_preferences.ndim == 1:
            source_preferences = np.asarray([source_preferences])
            target_preferences = np.asarray([target_preferences])

        if self.distance == loglikehood_coefficient:
            return self.distance(self.model.items_count(), \
                source_preferences, target_preferences) \
                if not source_preferences.shape[1] == 0 and \
                not target_preferences.shape[1] == 0 else np.array([[np.nan]])

        import pdb;pdb.set_trace()
        return self.distance(source_preferences, target_preferences) \
            if not source_preferences.shape[1] == 0 \
                and not target_preferences.shape[1] == 0 else np.array([[np.nan]])

    def get_similarities(self, source_id):
        return [(other_id, self.get_similarity(source_id, other_id))\
            for other_id, v in self.model]

    def __iter__(self):
        for source_id, preferences in self.model:
            yield source_id, self[source_id]


class ItemSimilarity(BaseSimiliarity):
    def __init__(self, model, distance, number=None):
        BaseSimiliarity.__init__(self, model, distance, number)

    def get_similarity(self, source_id, target_id):
        source_preferences = self.model.preference_for_item(source_id)
        target_preferences = self.model.preference_for_item(target_id)
        if self.model.has_preference_values():
            source_preferences, target_preferences = \
                find_common_elements(source_preferences, target_preferences)
        if source_preferences.ndim == 1 and target_preferences == 1:
            source_preferences = np.asarray([source_preferences])
            target_preferences = np.asarray([target_preferences])
        if self.distance == loglikehood_coefficient:
            return self.distance(self.model.items_count(), \
                source_preferences, target_preferences) \
                if not source_preferences.shape[1] == 0 and not \
                    target_preferences.shape[1] == 0 else np.array([[np.nan]])
        return self.distance(source_preferences, target_preferences) \
            if not source_preferences.shape[1] == 0 and not \
                target_preferences.shape[1] == 0 else np.array([[np.nan]])

    def get_similarities(self, source_id):
        return [(other_id, self.get_similarity(source_id, other_id)) for other_id in self.model.item_ids()]

    def __iter__(self):
        for item_id in self.model.item_ids():
            yield item_id, self[item_id]
