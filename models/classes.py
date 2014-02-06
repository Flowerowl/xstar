#encoding:utf-8
import logging

import numpy as np

from .base import BaseDataModel
from .exceptions import UserNotFoundError, ItemNotFoundError

logger = logging.getLogger("xstar")

class MatrixPreferenceDataModel(BaseDataModel):
    def __init__(self, dataset):
        BaseDataModel.__init__(self)
        self.dataset = dataset
        self.build_model()

    def __getitem__(self, user_id):
        return self.preference_from_user(user_id)

    def __iter__(self):
        for index, user in enumerate(self.user_ids()):
            yield user, self[user]

    def __len__(self):
        return self.index.shape

    def build_model(self):
        self._user_ids = np.asanyarray(self.dataset.keys())
        self._user_ids.sort()

        self._item_ids = []
        for item in self.dataset.itervalues():
            self._item_ids.extend(items.keys())

        self._item_ids = np.unique(np.array(self._item_ids))
        self._item_ids.sort()

        self.max_pref = -np.inf
        self.min_pref = np.inf

        logger.info("creating matrix for %d users and %d items" % (self._user_ids.size, self._item_ids.size))

        self.index = np.empty(shape=(self._user_ids, self._item_ids))
        for user_num, user_id in enumerate(self._user_ids):
            if user_num % 2 == 0:
                pass

    def user_ids(self):
        pass

    def item_ids(self):
        pass

    def preference_values_from_user(self, user_id):
        pass

    def preferences_from_user(self, user_id, order_by_id=True):
        pass

    def has_preference_values(self):
        pass

    def maximum_preference_value(self):
        pass

    def minimum_preference_value(self):
        pass

    def users_count(self):
        pass

    def items_count(self):
        pass

    def items_from_user(self):
        pass

    def preferences_for_item(self, item_id, order_by_id=True):
        pass

    def preference_value(self):
        pass

    def set_preference(self, user_id, item_id, value):
        pass

    def remove_preference(self, user_id, item_id):
        pass

    def __repr__(self):
        pass

    def _repr_matrix_(self, matrix):
        pass

    def __unicode__(self):
        pass

    def __str__(self):
        return unicode(self).encode('utf-8')
