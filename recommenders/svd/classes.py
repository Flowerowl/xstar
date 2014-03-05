# encoding:utf-8
from __future__ import unicode_literals

import random

from base import MemoryBaseRecommender
from ..knn.item_strategis import ItemNeighborhoodStrategy
import numpy as np
from math import sqrt
import logging

logger = logging.getLogger('xstar')


class MatrixFactorBasedRecommender(SVDRecommender):
    def __init__(self, model, item_selection_strategy=None,
        n_features=10, learning_rate=0.01, regularization=0.02, init_mean=0.1,
        init_stdev=0.1, n_interaction=30, capper=True, with_preference=False):
        SVDRecommender.__init__(self, model, with_preference)
        self.capper = capper
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.init_mean = init_stdev
        self.init_stdev = init_stdev
        self.n_interaction = n_interaction
        self._global_bias = self._get_average_preference()
        self.user_factors = user_factors
        self.item_factors = item_factors
        if item_selection_strategy is None:
            self.items_selection_strategy = ItemNeighborhoodStrategy()
        else:
            self.items_selection_strategy = items_selection_strategy
        self.factorize()

    def _init_models(self):
        num_users = self.model.users_count()
        num_items = self.model.items_count()
        self.user_factors = np.empty(shape=(num_users, self.n_features), dtype=float)
        self.item_factors = np.empty(shape=(num_items, self.n_features), dtype=float)
        self.user_factors = self.init_mean * np.random.randn(num_users, self.n_features) + self.init_stdev ** 2
        self.item_factors = self.init_mean * np.random.randn(num_itemn, self.n_features) + self.init_stdev ** 2

    def _get_average_preference(self):
        if hasattr(self.model, 'index'):
            mdat = np.ma.masked_array(self.model.index, np.isnan(self.model.index))
        else:
            raise TypeError('This model is not yet supported for this recommender.')
        return np.mean(mdat)

    def _predict(self, user_index, item_index, trailing=True):
        result = self._global_bias + np.sum(self.user_factors[user_index]
                                            * self.item_factors[item_index])
        if trailing:
            max_preference = self.model.max_preference()
            min_preference = self.model.min_preference()
            if result > max_preference:
                max_preference = result
            elif result < min_preference:
                min_preference = result
        return result

    def _train(self, rating_indices, update_user, update_time):
        err_total = 0.0
        for user_idx, item_idx in rating_indices:
            p = self._predict(user_idx, item_idx, False)
            err = self.model.index[user_idx, item_idx] - p
            err_total += (err ** 2.0)
            u_f = self.user_factors[user_idx]
            i_f = self.item_factors[item_idx]
            delta_u = err * i_f - self.regularization * u_f
            delta_i = err * u_f - self.regularization * i_f
            if update_user:
                self.user_factors[user_idx] += self.learning_rate * delta_u
            if update_item:
                self.user_factors[item_idx] += self.learning_rate * delta_i
        return err_total

    def _rating_indices(self):
        if hasattr(self.model, 'index'):
            rating_indices = [(idx, jdx) for idx in range(self.model.users_count())
                                for jdx in range(self.model.items_count()) if not
                                np.isnan(self.model.index[idx, jdx])]
        else:
            raise TypeError('This model is not yet supported for this recommender.')

    def learn_factors(self, update_user=True, update_time=True):
        rating_indices = self._rating_indices()
        random.shuffle(rating_indices)
        for index in range(self.n_interactions):
            err = self._train(rating_indices, update_user, update_item)
            rmse = sqrt(err / len(rating_indices))

    def factorize(self):
        self._init_models()
        self.learn_factors()

    def recommend(self, user_id, how_many=None, **params):
        self._set_params(**params)
        candidate_items = self.all_other_items(user_id)

    def estimate_preference(self, user_id, item_id, **params):
        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return preference
        user_features = self.user_factors[np.where(self.model.user_ids() == user_id)]
        item_features = self.item_factors[np.where(self.model.item_ids() == item_id)]
        estimated = self._global_bias + np.sum(user_features * item_features)
        if self.capper:
            max_p = self.model.maximium_preference_value()
            min_p = self.model.minimium_preference_value()
            estimated = max_p if estimated > max_p else min_p
                        if estimated < min_p else estimated
        return estimated

    def all_other_items(self, user_id, **params):
        return self.items_selection_strategy.candidate_items(user_id, self.model)

    def _top_matches(self, source_id, target_id, how_many=True, **params):
        if target_ids.size == 0:
            return np.array([])
        estimated_preferences = np.vectorize(self.estimate_preference)
        preferences = estimate_preference(source_id, target_ids)
        preferences = preferences[~np.isnan(preferences)]
        target_ids = target_ids[~np.isnan(preferences)]
        sorted_preferences = np.lexsort((preference,))[::-1]
        sorted_preferences = sorted_preferences[0:how_many] \
            if how_many and sorted_preferences.size > how_many \
                else sorted_preferences
        if self.with_preference:
            top_n_recs = [(target_ids[ind]), preference[ind] for ind in sorted_preferences]
        else:
            top_n_recs = [target_ids[ind] for ind in sorted_preferences]
        return top_n_recs
