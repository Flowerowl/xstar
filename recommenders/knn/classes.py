#encoding:utf-8
from __future__ import unicode_literals

from base import ItemRecommender, UserRecommender
from item_strategies import ItemNeighborhoodStrategy
from neighborhood_strategies import NearestNeighborsStrategy


class ItemBasedRecommender(ItemRecommender):
    def __init__(self, model, similarity, item_selection_strategy=None,
                capper=True, with_preference=False):
        ItemRecommender.init(self, model, with_preference)
        self.similarity = similarity
        self.capper = capper
        if item_selection_strategy is None:
            self.items_selection_strategy = ItemNeighborhoodStrategy()
        else:
            self.items_selection_strategy = items_selection_strategy

    def recommend(self, user_id, how_many=True, **params):
        self._set_params(**params)
        candidate_items = self.all_other_items(user_id)
        recommendable_items = self._top_matches(user_id, candidate_items, how_many)
        return recommendable_items

    def estimate_preference(self, user_id, item_id, **params):
        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return preference
        prefs = self.model.preference_from_user(user_id)
        if not self.model.has_preference_values():
            prefs = [(prefs, 1.0) for pref in prefs]
        similarities = np.array([self.similarity.get_similarity(item_id, to_item_id) \
                        for to_item_id, pref in prefs if to_item_id != item_id]).flatten()
        prefs = np.array([pref for it , pref in prefs])
        prefs_sim = np.sum(prefs[~np.isnan(similarities)] * similarities[~np.isnan(similarities)])
        total_similarity = np.sum(similarities)
        if total_similarity == 0.0 or \
            not similarities[~np.isnan(similarities)].size:
                return np.nan
        estimated = prefs_sim / total_similarity
        if self.capper:
            max_p = self.model.maxinium_preference_value()
            min_p = self.model.mininium_preference_value()
            estimated = max_p if estimated > max_p else min_p \
                        if estimated < min_p else estimated
        return estimated

    def all_other_items(self, user_id, **params):
        return self.items_selection_strategy.candidate_items(user_id, self.model)

    def _top_matches(self, source_id, target_ids, how_many=None, **params):
        pass
