# encoding:utf-8
from __future__ import unicode_literals

from ..base import MemoryBaseRecommender


class SVDRecommender(MemoryBaseRecommender):
    def factorize(self):
        raise NotImplementedError('ItemRecommender is an abstract class.')

    def train(self):
        raise NotImplementedError('ItemRecommender is an abstract class.')
