#encoding:utf-8
from xstar.models import MatrixPreferenceDataModel
from xstar.metrics import pearson_correlation
from xstar.similarities import UserSimilarity, ItemSimilarity
from xstar.recommenders.knn import UserBasedRecommender, ItemBasedRecommender


data = {
    'Marcel Caraciolo': {'Lady in the Water': 2.5, \
    'Snakes on a Plane': 3.5, \
    'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, \
    'The Night Listener': 3.0}, \
    'Paola Pow': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, \
    'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, \
    'You, Me and Dupree': 3.5}, \
    'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, \
    'Superman Returns': 3.5, 'The Night Listener': 4.0}, \
    'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, \
    'The Night Listener': 4.5, 'Superman Returns': 4.0, \
    'You, Me and Dupree': 2.5}, \
    'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
    'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, \
    'You, Me and Dupree': 2.0}, \
    'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
    'The Night Listener': 3.0, 'Superman Returns': 5.0, \
    'You, Me and Dupree': 3.5}, \
    'Penny Frewman': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0, \
    'Superman Returns':4.0}, \
    'Maria Gabriela': {}
}
model = MatrixPreferenceDataModel(data)
#similarity = UserSimilarity(model, pearson_correlation)
#recommender = UserBasedRecommender(model, similarity, with_preference=True)
#print recommender.recommend('Penny Frewman')
similarity = ItemSimilarity(model, pearson_correlation)
recommender = ItemBasedRecommender(model, similarity, with_preference=True)
print recommender.recommend('Just My Luck')
