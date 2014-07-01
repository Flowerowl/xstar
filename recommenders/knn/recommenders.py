# encoding:utf-8
from __future__ import unicode_literals

import numpy as np
from .base import ItemRecommender, UserRecommender
from .strategies import(
    ItemsNeighborhoodStrategy, NearestNeighborsStrategy
)


class ItemBasedRecommender(ItemRecommender):
    def __init__(self, model, similarity, items_selection_strategy=None,
                capper=True, with_preference=False):
        ItemRecommender.__init__(self, model, with_preference)
        self.similarity = similarity
        self.capper = capper
        if items_selection_strategy is None:
            self.items_selection_strategy = ItemsNeighborhoodStrategy()
        else:
            self.items_selection_strategy = items_selection_strategy

    def recommend(self, user_id, how_many=None, **params):
        self._set_params(**params)
        #所有可能的item ids
        candidate_items = self.all_other_items(user_id)
        recommendable_items = self._top_matches(user_id, \
                 candidate_items, how_many)
        return recommendable_items

    def estimate_preference(self, user_id, item_id, **params):
        #用户对此item的评分
        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return preference
        #用户所有评分[('Snakes on a Plane', 4.5),..]
        prefs = self.model.preferences_from_user(user_id)
        if not self.model.has_preference_values():
            prefs = [(pref, 1.0) for pref in prefs]

        #array([-0.33333333, -0.42289003, -0.48566186])
        similarities = np.array([self.similarity.get_similarity(item_id, to_item_id) \
            for to_item_id, pref in prefs if to_item_id != item_id]).flatten()
        #array([ 4.5,  4. ,  1. ])
        prefs = np.array([pref for it, pref in prefs])
        #-3.6772219907013066
        prefs_sim = np.sum(prefs[~np.isnan(similarities)] *
                             similarities[~np.isnan(similarities)])
        #-1.241885229201547
        total_similarity = np.sum(similarities)
        if total_similarity == 0.0 or \
           not similarities[~np.isnan(similarities)].size:
            return np.nan
        #2.9609998607242685
        estimated = prefs_sim / total_similarity
        if self.capper:
            #5.0
            max_p = self.model.maximum_preference_value()
            #1.0
            min_p = self.model.minimum_preference_value()
            #2.9609998607242685
            estimated = max_p if estimated > max_p else min_p \
                     if estimated < min_p else estimated
        return estimated

    def all_other_items(self, user_id, **params):
        return self.items_selection_strategy.candidate_items(user_id, \
                            self.model)

    def _top_matches(self, source_id, target_ids, how_many=None, **params):
        if target_ids.size == 0:
            return np.array([])
        estimate_preferences = np.vectorize(self.estimate_preference)
        #array([ 2.96099986,  3.61003107,  3.53139503])
        preferences = estimate_preferences(source_id, target_ids)
        #array([ 2.96099986,  3.61003107,  3.53139503])
        preference_values = preferences[~np.isnan(preferences)]
        target_ids = target_ids[~np.isnan(preferences)]
        sorted_preferences = np.lexsort((preference_values,))[::-1]
        #array([1, 2, 0])
        sorted_preferences = sorted_preferences[0:how_many] \
             if how_many and sorted_preferences.size > how_many \
                else sorted_preferences
        if self.with_preference:
            top_n_recs = [(target_ids[ind], \
                     preferences[ind]) for ind in sorted_preferences]
        else:
            top_n_recs = [target_ids[ind]
                 for ind in sorted_preferences]
        #['Lady in the Water', 'The Night Listener']
        return top_n_recs

    #def most_similar_items(self, item_id, how_many=None):
        #old_how_many = self.similarity.num_best
        #self.similarity.num_best = how_many + 1 \
                    #if how_many is not None else None
        #similarities = self.similarity[item_id]
        #self.similarity.num_best = old_how_many
        #return np.array([item for item, pref in similarities \
            #if item != item_id and not np.isnan(pref)])

    #def recommended_because(self, user_id, item_id, how_many=None, **params):
        #preferences = self.model.preferences_from_user(user_id)
        #if self.model.has_preference_values():
            #similarities = \
                #np.array([self.similarity.get_similarity(item_id, to_item_id) \
                    #for to_item_id, pref in preferences
                        #if to_item_id != item_id]).flatten()
            #prefs = np.array([pref for it, pref in preferences])
            #item_ids = np.array([it for it, pref in preferences])
        #else:
            #similarities = \
                #np.array([self.similarity.get_similarity(item_id, to_item_id) \
                #for to_item_id in preferences
                    #if to_item_id != item_id]).flatten()
            #prefs = np.array([1.0 for it in preferences])
            #item_ids = np.array(preferences)
        #scores = prefs[~np.isnan(similarities)] * \
             #(1.0 + similarities[~np.isnan(similarities)])
        #sorted_preferences = np.lexsort((scores,))[::-1]
        #sorted_preferences = sorted_preferences[0:how_many] \
             #if how_many and sorted_preferences.size > how_many \
                 #else sorted_preferences
        #if self.with_preference:
            #top_n_recs = [(item_ids[ind], \
                     #prefs[ind]) for ind in sorted_preferences]
        #else:
            #top_n_recs = [item_ids[ind]
                 #for ind in sorted_preferences]
        #return top_n_recs


class UserBasedRecommender(UserRecommender):
    def __init__(self, model, similarity, neighborhood_strategy=None,
                capper=True, with_preference=False):
        UserRecommender.__init__(self, model, with_preference)
        self.similarity = similarity
        self.capper = capper
        if neighborhood_strategy is None:
            self.neighborhood_strategy = NearestNeighborsStrategy()
        else:
            self.neighborhood_strategy = neighborhood_strategy

    def all_other_items(self, user_id, **params):
        n_similarity = params.pop('n_similarity', 'user_similarity')
        distance = params.pop('distance', self.similarity.distance)
        nhood_size = params.pop('nhood_size', None)
        #[('Marcel Caraciolo', 0.99124070716193036), ('Steve Gates', 0.92447345164190498), ('Lorena Abreu', 0.89340514744156441), ('Sheldom', 0.66284898035987017), ('Paola Pow', 0.3812464258315118), ('Leopoldo Pires', -0.99999999999999978)]
        nearest_neighbors = self.neighborhood_strategy.user_neighborhood(user_id,
                self.model, n_similarity, distance, nhood_size, **params)
        #本用户评价过的物品
        items_from_user_id = self.model.items_from_user(user_id)
        possible_items = []
        for to_user_id in nearest_neighbors:
            possible_items.extend(self.model.items_from_user(to_user_id))
        # 邻居用户评价过的物品
        possible_items = np.unique(np.array(possible_items).flatten())
        # 返回所有其他用户评价过的物品（不包含自身评价过的）
        return np.setdiff1d(possible_items, items_from_user_id)

    def estimate_preference(self, user_id, item_id, **params):
        #import pdb;pdb.set_trace()
        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return preference
        n_similarity = params.pop('n_similarity', 'user_similarity')
        distance = params.pop('distance', self.similarity.distance)
        nhood_size = params.pop('nhood_size', None)
        #['Marcel Caraciolo', 'Steve Gates', 'Lorena Abreu', 'Sheldom', 'Paola Pow']
        nearest_neighbors = self.neighborhood_strategy.user_neighborhood(user_id,
                self.model, n_similarity, distance, nhood_size, **params)
        preference = 0.0
        total_similarity = 0.0
        #array([ 0.99124071,  0.92447345,  0.89340515,  0.66284898,  0.38124643])
        similarities = np.array([self.similarity.get_similarity(user_id, to_user_id)
                for to_user_id in nearest_neighbors]).flatten()
        #array([ 3. ,  2. ,  3. ,  nan,  1.5])
        prefs = np.array([self.model.preference_value(to_user_id, item_id)
                 for to_user_id in nearest_neighbors])
        #array([ 3. ,  2. ,  3. ,  1.5])
        prefs = prefs[~np.isnan(prefs)]
        #array([ 0.99124071,  0.92447345,  0.89340515,  0.66284898])
        similarities = similarities[~np.isnan(prefs)]
        #8.4971579376340998
        prefs_sim = np.sum(prefs[~np.isnan(similarities)] *
                             similarities[~np.isnan(similarities)])
        #3.4719682866052697
        total_similarity = np.sum(similarities)
        if total_similarity == 0.0 or \
           not similarities[~np.isnan(similarities)].size:
            return np.nan
        #2.4473604699719846
        estimated = prefs_sim / total_similarity
        if self.capper:
            max_p = self.model.maximum_preference_value()
            min_p = self.model.minimum_preference_value()
            estimated = max_p if estimated > max_p else min_p \
                     if estimated < min_p else estimated
        return estimated

    def recommend(self, user_id, how_many=None, **params):
        self._set_params(**params)
        # 所有邻居用户评价过的物品
        candidate_items = self.all_other_items(user_id, **params)
        recommendable_items = self._top_matches(user_id, \
                 candidate_items, how_many)
        return recommendable_items

    def _top_matches(self, source_id, target_ids, how_many=None, **params):
        if target_ids.size == 0:
            return np.array([])
        #关于np.vectorize
        #ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html
        estimate_preferences = np.vectorize(self.estimate_preference)
        # source_id是用户ID，target_ids未评价过的推荐物品
        preferences = estimate_preferences(source_id, target_ids)
        preference_values = preferences[~np.isnan(preferences)]
        target_ids = target_ids[~np.isnan(preferences)]
        sorted_preferences = np.lexsort((preference_values,))[::-1]
        sorted_preferences = sorted_preferences[0:how_many] \
             if how_many and sorted_preferences.size > how_many \
                else sorted_preferences
        if self.with_preference:
            top_n_recs = [(target_ids[ind], \
                     preferences[ind]) for ind in sorted_preferences]
        else:
            top_n_recs = [target_ids[ind]
                 for ind in sorted_preferences]
        return top_n_recs

    #def most_similar_users(self, user_id, how_many=None):
        #old_how_many = self.similarity.num_best
        #self.similarity.num_best = how_many + 1 \
                    #if how_many is not None else None
        #similarities = self.similarity[user_id]
        #self.similarity.num_best = old_how_many
        #return np.array([to_user_id for to_user_id, pref in similarities \
            #if user_id != to_user_id and not np.isnan(pref)])

    #def recommended_because(self, user_id, item_id, how_many=None, **params):
        #preferences = self.model.preferences_for_item(item_id)
        #if self.model.has_preference_values():
            #similarities = \
                #np.array([self.similarity.get_similarity(user_id, to_user_id) \
                    #for to_user_id, pref in preferences
                        #if to_user_id != user_id]).flatten()
            #prefs = np.array([pref for it, pref in preferences])
            #user_ids = np.array([usr for usr, pref in preferences])
        #else:
            #similarities = \
                #np.array([self.similarity.get_similarity(user_id, to_user_id) \
                #for to_user_id in preferences
                    #if to_user_id != user_id]).flatten()
            #prefs = np.array([1.0 for it in preferences])
            #user_ids = np.array(preferences)
        #scores = prefs[~np.isnan(similarities)] * \
             #(1.0 + similarities[~np.isnan(similarities)])
        #sorted_preferences = np.lexsort((scores,))[::-1]
        #sorted_preferences = sorted_preferences[0:how_many] \
             #if how_many and sorted_preferences.size > how_many \
                 #else sorted_preferences
        #if self.with_preference:
            #top_n_recs = [(user_ids[ind], \
                     #prefs[ind]) for ind in sorted_preferences]
        #else:
            #top_n_recs = [user_ids[ind]
                 #for ind in sorted_preferences]
        #return top_n_recs
