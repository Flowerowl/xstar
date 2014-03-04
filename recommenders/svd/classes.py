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
