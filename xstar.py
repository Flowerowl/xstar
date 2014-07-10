#encoding: utf-8
from __future__ import unicode_literals

import numpy as np
import scipy.spatial.distance as ssd
from scikits.learn.base import BaseEstimator


##################
#MODEL
##################
class UserNotFoundError(Exception):
    pass


class DataModel(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.build_model()

    def __getitem__(self, user_id):
        return self.preferences_from_user(user_id)

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
            self._item_ids.extend(item.keys())

        self._item_ids = np.unique(np.array(self._item_ids))
        self._item_ids.sort()

        self.max_pref = -np.inf
        self.min_pref = np.inf

        self.index = np.empty(shape=(self._user_ids.size, self._item_ids.size))
        for user_num, user_id in enumerate(self._user_ids):
            for item_num, item_id in enumerate(self._item_ids):
                r = self.dataset[user_id].get(item_id, np.NaN)
                self.index[user_num, item_num] = r
        if self.index.size:
            self.max_pref = np.nanmax(self.index)
            self.min_pref = np.nanmin(self.index)

    def user_ids(self):
        return self._user_ids

    def item_ids(self):
        return self._item_ids

    def preference_values_from_user(self, user_id):
        user_id_loc = np.where(self._user_ids == user_id)
        if not user_id_loc[0].size:
            raise UserNotFoundError
        preferences = self.index[user_id_loc]
        return preferences

    def preferences_from_user(self, user_id, order_by_id=True):
        preferences = self.preference_values_from_user(user_id)
        data = zip(self._item_ids, preferences.flatten())
        if order_by_id:
            return [(item_id, preference) for item_id, preference in data \
                        if not np.isnan(preference)]
        else:
            return sorted([(item_id, preference) for item_id, preference in data \
                            if not isnan(preference)], key=lambda item: - item[1])

    def has_preference_values(self):
        return True

    def maximum_preference_value(self):
        return self.max_pref

    def minimum_preference_value(self):
        return self.min_pref

    def users_count(self):
        return self._user_ids.size

    def items_count(self):
        return self._item_ids.size

    def items_from_user(self, user_id):
        preferences = self.preferences_from_user(user_id)
        return [key for key, value in preferences]

    def preferences_for_item(self, item_id, order_by_id=True):
        item_id_loc = np.where(self._item_ids == item_id)
        if not item_id_loc[0].size:
            raise ItemNotFoundError('Item not Found')
        preferences = self.index[:, item_id_loc]
        data = zip(self._user_ids, preferences.flatten())
        if order_by_id:
            return [(user_id, preference) for user_id, preference in data\
                        if not np.isnan(preference)]
        else:
            return sorted([(user_id, preference) for user_id, preference in data \
                            if not np.isnan(preference)], key=lambda user: - user[1])

    def preference_value(self, user_id, item_id):
        item_id_loc = np.where(self._item_ids == item_id)
        user_id_loc = np.where(self._user_ids == user_id)
        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')
        if not item_id_loc[0].size:
            raise ItemNotFoundError('item_id in the model not found')
        return self.index[user_id_loc, item_id_loc].flatten()[0]

    def set_preference(self, user_id, item_id, value):
        user_id_loc = np.where(self._user_ids == user_id)
        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')

    def remove_preference(self, user_id, item_id):
        user_id_loc = np.where(self._user_ids == user_id)
        item_id_loc = np.where(self._item_ids == item_id)
        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')
        if not item_id_loc[0].size:
            raise ItemNotFoundError('item_id in the model not found')
        del self.dataset[user_id][item_id]
        self.build_model()

    def __repr__(self):
        return '<MatrixPreferenceDataModel (%d by %d)>' % (self.index.shape[0],
                    self.index.shape[1])

    def _repr_matrix(self, matrix):
        s = ''
        cell_width = 11
        shape = matrix.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    s += '---'.center(cell_width)
                else:
                    exp = np.log(abs(v))
                    if abs(exp) <= 4:
                        if exp < 0:
                            s += ('%9.6f' % v).ljust(cell_width)
                        else:
                            s += ('%9.*f' % (6, v)).ljust(cell_width)
                    else:
                        s += ('%9.2e' % v).ljust(cell_width)
            s += '\n'
        return s[:-1]

    def __unicode__(self):
        matrix = self._repr_matrix(self.index[:20, :5])
        lines = matrix.split('\n')
        headers = [repr(self)[1:-1]]
        if self._item_ids.size:
            col_headers = [('%-8s' % unicode(item)[:8]) for item in self._item_ids[:5]]
            headers.append(' ' + (' '.join(col_headers)))
        if self._user_ids.size:
            for (i, line) in enumerate(lines):
                lines[i] = ('%-8s' % unicode(self._user_ids[i])[:8]) + line
            for (i, line) in enumerate(headers):
                if i > 0:
                    headers[i] = ' ' * 8 + line
        lines = headers + lines
        if self.index.shape[1] > 5 and self.index.shape[0] > 0:
            lines[1] += '...'
        if self.index.shape[0] > 20:
            lines.append('...')
        return '\n'.join(line.rstrip() for line in lines)

    def __str__(self):
        return unicode(self).encode('utf-8')


##################
#DISTANCE
##################
def euclidean_distances(X, Y, squared=False, inverse=True):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y metrics")
    if squared:
        return ssd.cdist(X, Y, 'sqeuclidean')
    XY = ssd.cdist(X, Y)
    return np.divide(1.0, (1.0+XY)) if inverse else XY

euclidean_distances = euclidean_distances


def pearson_correlation(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y metrics")
    XY = ssd.cdist(X, Y, 'correlation', 2)
    return 1 - XY


##################
#SIMILARITY
##################
def find_common_elements(source_preferences, target_preferences):
    src = dict(source_preferences)
    tgt = dict(target_preferences)

    #找出交集id
    inter = np.intersect1d(src.keys(), tgt.keys())

    common_preferences = zip(*[(src[item], tgt[item]) for item in inter \
        if not np.isnan(src[item]) and not np.isnan(tgt[item])])
    if common_preferences:
        return np.asarray([common_preferences[0]]), np.asarray([common_preferences[1]])
    else:
        return np.asarray([[]]), np.asarray([[]])


class UserSimilarity(object):
    def __init__(self, model, distance, num_best=None, number=None):
        self.model = model
        self.distance = distance
        self.num_best = num_best
        self.number = number

    def get_similarity(self, source_id, target_id):
        #import pdb;pdb.set_trace()
        #本用户的物品评分
        source_preferences = self.model.preferences_from_user(source_id)
        #目标用户的物品评分
        target_preferences = self.model.preferences_from_user(target_id)

        if self.model.has_preference_values():
            source_preferences, target_preferences = \
                find_common_elements(source_preferences, target_preferences)

        if source_preferences.ndim == 1 and target_preferences.ndim == 1:
            source_preferences = np.asarray([source_preferences])
            target_preferences = np.asarray([target_preferences])

        return self.distance(source_preferences, target_preferences) \
            if not source_preferences.shape[1] == 0 \
                and not target_preferences.shape[1] == 0 else np.array([[np.nan]])

    def get_similarities(self, source_id):
        return[(other_id, self.get_similarity(source_id, other_id))  for other_id, v in self.model]

    def __iter__(self):
        for source_id, preferences in self.model:
            yield source_id, self[source_id]

    def __getitem__(self, source_id):
        similar_items = self.get_similarities(source_id)
        tops = sorted(similar_items, key=lambda x: -x[1])
        if similar_items:
            item_ids, preferences = zip(*similar_items)
            preferences = np.array(preferences).flatten()
            item_ids = np.array(item_ids).flatten()
            sorted_prefs = np.argsort(-preferences)
            tops = zip(item_ids[sorted_prefs], preferences[sorted_prefs])
        return tops[:self.num_best] if self.num_best is not None else tops


##################
#KNN
##################
class NearestNeighbors(object):
    def __init__(self):
        self.similarity = None

    def _sampling(self, data_model, sampling_rate):
        return data_model

    def _set_similarity(self, data_model, similarity, distance, nhood_size):
        if not isinstance(self.similarity, UserSimilarity) \
             or not distance == self.similarity.distance:
            nhood_size = nhood_size if not nhood_size else nhood_size + 1
            self.similarity = UserSimilarity(data_model, distance, nhood_size)

    def user_neighborhood(self, user_id, data_model, n_similarity='user_similarity',
             distance=None, nhood_size=None, **params):
        minimal_similarity = params.get('minimal_similarity', 0.0)
        sampling_rate = params.get('sampling_rate', 1.0)

        data_model = self._sampling(data_model, sampling_rate)
        if distance is None:
            distance = euclidean_distances
        if n_similarity == 'user_similarity':
            self._set_similarity(data_model, n_similarity, distance, nhood_size)
        else:
            raise ValueError('similarity argument must be user_similarity')
        #[('Marcel Caraciolo', 0.99124070716193036), ('Steve Gates', 0.92447345164190498), ('Lorena Abreu', 0.89340514744156441), ('Sheldom', 0.66284898035987017), ('Paola Pow', 0.3812464258315118), ('Leopoldo Pires', -0.99999999999999978)]
        #import pdb;pdb.set_trace()
        neighborhood = [to_user_id for to_user_id, score in self.similarity[user_id] \
                           if not np.isnan(score) and score >= minimal_similarity and \
                           user_id != to_user_id]
        return neighborhood


class UserBasedRecommender(BaseEstimator):
    def __init__(self, model, similarity, neighborhood=None,
                capper=True, with_preference=False):
        self.model = model
        self.similarity = similarity
        self.capper = capper
        self.with_preference = with_preference
        if neighborhood ==  None:
            self.neighborhood = NearestNeighbors()

    def all_other_items(self, user_id, **params):
        n_similarity = params.pop('n_similarity', 'user_similarity')
        distance = params.pop('distance', self.similarity.distance)
        nhood_size = params.pop('nhood_size', None)
        #[('Marcel Caraciolo', 0.99124070716193036), ('Steve Gates', 0.92447345164190498), ('Lorena Abreu', 0.89340514744156441), ('Sheldom', 0.66284898035987017), ('Paola Pow', 0.3812464258315118), ('Leopoldo Pires', -0.99999999999999978)]
        nearest_neighbors = self.neighborhood.user_neighborhood(user_id,
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
        nearest_neighbors = self.neighborhood.user_neighborhood(user_id,
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
