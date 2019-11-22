import numpy as np 

class Comparer():
	""" given the features for two ids, predicts which should be 
	ranked higher
	"""

	def __init__(self, desc=None):
		""" desc is comparer description to use in place of default """
		self.desc = desc
		pass

	def __str__(self):
		if self.desc is None:
			return "Comparer"
		return self.desc

	def fit(self, id1_feats, id2_feats, labels):
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
			labels: array of length n_samples which is 1 if id1 is ranked higher
				than id2 and 0 otherwise

		Returns:
			fit self
		"""
		return self

	def predict(self, id1_feats, id2_feats):
		""" Predicts class of relationship between id1 and id2

		Predicts 1 or 0. 1 if id1 is predicted ranked higher, else 0

		This implementation should not actually be used, always predicts 1. 
		Purpose is only as function prototype.

		Args:
			id1_feats: array of shape [n_samples, n_features] which contains 
				the features for the id1 
			ids_feats: array of shape [n_samples, n_features] which contains 
				the features for the id2

		Returns:
			array of length n_samples were each value is the predicted 
				relationship between the ids at that index
		"""
		return np.ones(len(id1_feats))

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
		return np.ones(len(id1_feats)) * .5


class SklearnClassifierComparer(Comparer):
	""" given the features for two ids, predicts which should be 
	ranked higher using sklearn classifier
	"""

	def __init__(self, classifier, desc=None):
		""" desc is comparer description to use in place of sklearn str """
		self.classifier = classifier
		self.desc = desc

	def __str__(self):
		if self.desc is None:
			return "SklC: {}".format(str(self.classifier))
		return "SklC: {}".format(self.desc)

	def fit(self, id1_feats, id2_feats, labels):
		""" fits underlying model to predict 1 if id1 should be ranked higher 
		than id2 or 0 if the reverse is true. label is 1 or 0.

		Args:

		Returns:
			fit self
		"""
		X = np.concatenate((id1_feats, id2_feats), axis=1)
		self.classifier = self.classifier.fit(X, labels)
		return self

	def predict(self, id1_feats, id2_feats):
		""" Predicts class of relationship between id1 and id2

		Predicts 1 or 0. 1 if id1 is predicted ranked higher, else 0

		This implementation should not actually be used, always predicts 1. 
		Purpose is only as function prototype.

		Args:
			id1_feats: array of shape [n_samples, n_features] which contains 
				the features for the id1 
			ids_feats: array of shape [n_samples, n_features] which contains 
				the features for the id2

		Returns:
			array of length n_samples were each value is the predicted 
				relationship between the ids at that index
		"""
		X = np.concatenate((id1_feats, id2_feats), axis=1)
		return self.classifier.predict(X)

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
		X = np.concatenate((id1_feats, id2_feats), axis=1)
		return self.classifier.predict_proba(X)[:, 1]


class DiffSklearnClassifierComparer(SklearnClassifierComparer):
	""" given the features for two ids, predicts which should be 
	ranked higher using sklearn classifier. In addition to using the feature
	vectors for the two being compared, also appends the difference between 
	the who peoples feature vectors
	"""

	def __str__(self):
		if self.desc is None:
			return "DiffSklC: {}".format(str(self.classifier))
		return "DiffSklC: {}".format(self.desc)

	def fit(self, id1_feats, id2_feats, labels):
		""" fits underlying model to predict 1 if id1 should be ranked higher 
		than id2 or 0 if the reverse is true. label is 1 or 0.

		Args:

		Returns:
			fit self
		"""
		id1_feats = np.asarray(id1_feats)
		id2_feats = np.asarray(id2_feats)
		feat_difs = id1_feats - id2_feats

		X = np.concatenate((id1_feats, id2_feats, feat_difs), axis=1)
		self.classifier = self.classifier.fit(X, labels)
		return self

	def predict(self, id1_feats, id2_feats):
		""" Predicts class of relationship between id1 and id2

		Predicts 1 or 0. 1 if id1 is predicted ranked higher, else 0

		This implementation should not actually be used, always predicts 1. 
		Purpose is only as function prototype.

		Args:
			id1_feats: array of shape [n_samples, n_features] which contains 
				the features for the id1 
			ids_feats: array of shape [n_samples, n_features] which contains 
				the features for the id2

		Returns:
			array of length n_samples were each value is the predicted 
				relationship between the ids at that index
		"""
		id1_feats = np.asarray(id1_feats)
		id2_feats = np.asarray(id2_feats)
		feat_difs = id1_feats - id2_feats

		X = np.concatenate((id1_feats, id2_feats, feat_difs), axis=1)
		return self.classifier.predict(X)

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
		id1_feats = np.asarray(id1_feats)
		id2_feats = np.asarray(id2_feats)
		feat_difs = id1_feats - id2_feats

		X = np.concatenate((id1_feats, id2_feats, feat_difs), axis=1)
		return self.classifier.predict_proba(X)[:, 1]

class OnlyDiffSklearnClassifierComparer(SklearnClassifierComparer):
	""" given the features for two ids, predicts which should be 
	ranked higher using sklearn classifier. Feature vector is difference of
	id1 and id2 feats
	"""
	def __str__(self):
		if self.desc is None:
			return "OnlyDiffSklC: {}".format(str(self.classifier))
		return "OnlyDiffSklC: {}".format(self.desc)

	def fit(self, id1_feats, id2_feats, labels):
		""" fits underlying model to predict 1 if id1 should be ranked higher 
		than id2 or 0 if the reverse is true. label is 1 or 0.

		Args:

		Returns:
			fit self
		"""
		id1_feats = np.asarray(id1_feats)
		id2_feats = np.asarray(id2_feats)
		feat_difs = id1_feats - id2_feats

		self.classifier = self.classifier.fit(feat_difs, labels)
		return self

	def predict(self, id1_feats, id2_feats):
		""" Predicts class of relationship between id1 and id2

		Predicts 1 or 0. 1 if id1 is predicted ranked higher, else 0

		This implementation should not actually be used, always predicts 1. 
		Purpose is only as function prototype.

		Args:
			id1_feats: array of shape [n_samples, n_features] which contains 
				the features for the id1 
			ids_feats: array of shape [n_samples, n_features] which contains 
				the features for the id2

		Returns:
			array of length n_samples were each value is the predicted 
				relationship between the ids at that index
		"""
		id1_feats = np.asarray(id1_feats)
		id2_feats = np.asarray(id2_feats)
		feat_difs = id1_feats - id2_feats

		return self.classifier.predict(feat_difs)

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
		id1_feats = np.asarray(id1_feats)
		id2_feats = np.asarray(id2_feats)
		feat_difs = id1_feats - id2_feats

		return self.classifier.predict_proba(feat_difs)[:, 1]
