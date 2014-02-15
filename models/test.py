#encoding:utf-8
from __future__ import unicode_literals, print_function

from .classes import MatrixPreferenceDataModel


movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
            'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
            'The Night Listener': 3.0},
            'Luciana Nunes': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
            'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
            'You, Me and Dupree': 3.5},
            'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
            'Superman Returns': 3.5, 'The Night Listener': 4.0},
            'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
            'The Night Listener': 4.5, 'Superman Returns': 4.0,
            'You, Me and Dupree': 2.5},
            'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
            'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
            'You, Me and Dupree': 2.0},
            'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
            'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
            'Penny Frewman': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0},
            'Maria Gabriela': {}}


model = MatrixPreferenceDataModel(movies)
print ('user_ids:', model.user_ids())
print ('item_ids:', model.item_ids())
print ('preferences_from_user("Marcel Caraciolo"):', model.preferences_from_user('Leopoldo Pires'))
print ('has_preference_values:', model.has_preference_values())
print ('maximum_preference_value:', model.maximum_preference_value())
print ('minimum_preference_value:', model.minimum_preference_value())
print ('users_count:', model.users_count())
print ('items_count:', model.items_count())
print ('items_from_user("Marcel Caraciolo"):', model.items_from_user('Marcel Caraciolo'))
print ('preferences_for_item("Lady in the Water"):', model.preferences_for_item('Lady in the Water'))
print ('preference_value("Marcel Caraciolo", "Lady in the Water")', model.preference_value('Marcel Caraciolo', 'Lady in the Water'))
print ('set_preference("Marcel Caraciolo", "Lady in the Water", 5)', model.set_preference('Marcel Caraciolo', 'Lady in the Water', 5))
