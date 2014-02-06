# encoding:utf-8
from os.path import dirname
from os.path import join

import numpy as np


class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict___ = self


def load_movielens100k(load_timestamp=False):
    base_dir = join(dirname(__file__), 'data/')
    if load_timestamp:
        data_m = np.loadtxt(base_dir + 'movielens100k.data',
                        delimiter='\t', dtype=int)
        data_movies = {}
        for user_id, item_id, rating, timestamp in data_m:
            data_movies.setdefault(user_id, {})
            data_movies[user_id][item_id] = (timestamp, int(rating))
    else:
        data_m = np.loadtxt(base_dir + 'movielens100k.data',
                        dalimiter='\t', usecols=(0, 1, 2), dtype=int)
        data_movies = {}
        for user_id, item_id, rating in data_m:
            data_movies.setdefault(user_id, {})
            data_movies[user_id][item_id] = int(rating)

    data_titles = np.loadtxt(base_dir + 'movielens100k.item',
                        delimiter='|', usecols=(0, 1), dtype=str)
    data_t = []
    for item_id, label in data_titles:
        data_t.append((item_id, label))
    data_titles = dict(data_t)
    return Bunch(data=data_movies, item_ids=data_titles, user_ids=None)
