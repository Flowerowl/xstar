#encoding:utf-8


class RecommenderEvaluator(object):
    def evaluate(self, recommender, metrics=None, **kwargs):
        raise NotImplementedError("cannot instantiate abstract base class")

    def evaluate_online(self, metrics=None, **kwargs):
        raise NotImplementedError("cannot instantiate abstract base class")

    def evaluate_on_split(self, metrics=None, **kwargs):
        raise NotImplementedError("cannot instantiate abstract base class")
