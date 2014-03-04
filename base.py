# encoding:utf-8
from __future__ import unicode_literals

from scikits.learn.base import BaseEstimator


class BaseRecommend(BaseEstimator):
    def __init__(self, model, with_preference=False):
        self.model = model
        self.with_preference = with_preference

    def recommend(self, user_id, how_many, **params):
        raise NotImplementedError('BaseRecommender is an abstract class.')

    def estimate_preference(self, user_id, item_id, **params):
        raise NotImplementedError('BaseRecommend is an abstract class.')

    def all_other_items(self, user_id, **params):
        raise NotImplementedError('BaseRecommend is an abstract class.')

    def set_preference(self, user_id, item_id, value):
        self.model.set_preference(user_id, item_id, value)

    def remove_preference(self, user_id, item_id):
        self.model.remove_preference(user_id, item_id)
