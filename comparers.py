import numpy as np 

class Comparer():
	""" given the features for two ids, predicts which should be 
	ranked higher
	"""

	def __init__(self):
		pass

	def fit(self, id1_feats, id2_feats, label):
		""" fits underlying model to predict 1 if id1 should be ranked higher 
		than id2 or 0 if the reverse is true. label is this 1 or 0.

		Args:
			id1_feats: array of shape [n_samples, n_features] which contains 
				the features for the id1 in this comparison where a higher
				ranked id1 is labeled 1
			ids_feats: array of shape [n_samples, n_features] which contains 
				the features for the id2 in this comparison where a higher
				ranked id2 is labeled 0. n_features should be the same as 
				id1_feats n_features
			label: array of length n_samples which is 1 if id1 is ranked higher
				than id2 and 0 otherwise

		Returns:
			fit self
		"""
		pass

	def predict(self, id1_feats, id2_feats):
		""" Predicts class of relationship between id1 and id2

		Predicts 1 or 0. 1 if id1 is predicted ranked higher and 0 if id2 is

		This implementation should not actually be used, always predicts 1. 
		Purpose is only as function prototype.

		Args:
			id1_feats: array of shape [n_samples, n_features] which contains 
				the features for the id1 
			ids_feats: array of shape [n_samples, n_features] which contains 
				the features for the id2
			label: array of length n_samples which is 1 if id1 is ranked higher
				than id2 and 0 otherwise

		Returns:
			array of length n_samples were each value is the predicted 
				relationship between the ids at that index
		"""
		return 1

	def predict_proba(self, id1_feats, id2_feats):
		""" Predicts probability id1 is ranked higher than id2

		This implementation always predicts .5

		Args:
			id1_feats: array of shape [n_samples, n_features] which contains 
				the features for the id1 
			ids_feats: array of shape [n_samples, n_features] which contains 
				the features for the id2

		Returns:
			array of length n_samples were each value is the probability that
				the id1 at that index is ranked higher than the id2 at that 
				index
		"""
		return .5
