#encoding:utf-8
from __future__ import unicode_literals, absolute_import


class BaseDataModel(object):

    def user_ids(self):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def item_ids(self):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preferences_values_from_user(self, user_id, order_by_id=True):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preferences_from_user(self, user_id, order_by_id=True):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preferences_for_item(self, item_id, order_by_id=True):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preference_value(self, user_id, item_id):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preference_item(self, user_id, item_id):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def item_from_user(self, user_id):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def users_count(self):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def items_count(self):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def set_preference(self, user_id, item_id, value=None):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def remove_preference(self, user_id, item_id):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def has_preference_value(self):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def maximum_preference_value(self):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def minimum_preference_value(self):
        raise NotImplementedError("cannot instantiate Abstract Base Class")
