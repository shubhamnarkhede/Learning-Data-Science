import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it 

data = fetch_movielens(min_rating=4.0)

#print training and testing data
print(repe(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss='warp')

#train model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):

	#number of users and movies in training data
	n_users, n_items = data['train'].shape

	#generate recommendation for each user we input
	for user_id in user_ids:

		#movies they already like
		know_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]