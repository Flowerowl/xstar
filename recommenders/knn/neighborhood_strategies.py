from base import BaseUserNeighborhoodStrategy
import numpy as np
from ...similarities.basic_similarities import UserSimilarity
from ...metrics.pairwise import euclidean_distances


class AllNeighborsStrategy(BaseUserNeighborhoodStrategy):
    def user_neighborhood(self, user_id, data_model, similarity='user_similarity',
        distance=None, nhood_size=None, **params):
        user_ids = data_model.user_ids()
        return user_ids[user_ids != user_id] if user_ids.size else user_ids


class NearestNeighborsStrategy(BaseUserNeighborhoodStrategy):
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
        neighborhood = [to_user_id for to_user_id, score in self.similarity[user_id] \
                           if not np.isnan(score) and score >= minimal_similarity and user_id != to_user_id]
        return neighborhood