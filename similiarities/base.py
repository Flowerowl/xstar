#encoding:utf-8
from __future__ import unicode_literals

import numpy as np


class BaseSimiliarity(object):
    def __init__(self, model, distance, number=None):
        self.model = model
        self.distance = distance
        self._set_number(number)

    def _set_number(self, number):
        self.number = number

    def get_similarity(self, source_id, target_id):
        raise NotImplementedError('cannot instantiate Abstract Base Class')

    def get_similarities(self, source_id):
        raise NotImplementedError('cannot instantiate Abstract Base Class')

    def __getitem__(self, source_id):
        similiar_items = self.get_similiarities(source_id)
        tops= sorted(similiar_items, key = lambda x: -x[1])
        if similiar_items:
            item_ids, preferences = zip(*similiar_items)
            item_ids = np.array(item_ids).flatten()
            sorted_prefs = np.argsort(-preferences)
            tops = zip(item_ids[sorted_prefs], preferences[sorted_prefs])
        return tops[:self.number] if self.number is not None else tops
