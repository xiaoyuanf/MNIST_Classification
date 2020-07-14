"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

	def __init__(self, k):
		self.k = k

	def fit(self, X, y):
		self.X = X # just memorize the trianing data
		self.y = y 

	def predict(self, Xtest):
		pred_labels=[]
		distance=utils.euclidean_dist_squared(Xtest, self.X)#row of X_test * row of X

		for row in range(0, distance.shape[0]):
			distance_per_pt=distance[row, :]
			np.sort(distance_per_pt)[:self.k]
			#get the index of nearest k neighbours
			nn_index=np.argsort(distance_per_pt)[:self.k]

			#get the labels of the k neighbours
			knn_labels=[self.y[i] for i in nn_index]
			#get most common labels
			labels_mode=utils.mode(np.array(knn_labels))

			pred_labels.append(labels_mode)

		return np.array(pred_labels)


