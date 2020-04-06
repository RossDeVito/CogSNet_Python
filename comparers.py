import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

from rankers import *

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
	"""
	Given the features for two ids, predicts which should be 
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
		""" 
		Fits underlying model to predict 1 if id1 should be ranked higher 
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
	""" 
	Given the features for two ids, predicts which should be 
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


""" Time Series Comparers """

class TimeSeriesComparer(Comparer):
	"""
	Uses Keras model to compare time series
	"""

	def __init__(self, model, desc=None, scaler=StandardScaler(),
				 batch_size=None, epochs=10, callbacks=None, validation_split=.1,
				 verbose=0, fit_verbose=1, n_workers=1):
		""" 
		verbose 3 or higher will save plots of model fitting history
		"""
		self.model = model
		self.desc = desc
		self.scaler = scaler
		self.batch_size = batch_size
		self.epochs = epochs
		self.callbacks = callbacks
		self.verbose = verbose
		self.fit_verbose = fit_verbose
		self.validation_split = validation_split
		self.n_workers = n_workers

	def __str__(self):
		if self.desc is None:
			return "TSC: {}".format(str(self.model))
		return "TSC: {}".format(self.desc)

	def fit(self, feats, labels):
		"""
		
		Args:

		Returns:
			fit self
		"""

		if self.verbose > 0:
			print("Fitting scaler")

		n_channels = feats.shape[1]

		feats_by_channel = feats.transpose([1, 2, 0]).reshape([n_channels, -1])
		self.scaler.fit(feats_by_channel.T)

		if self.verbose > 0:
			print("Scaling data")

		scaled_data = []
		for i in range(feats.shape[0]):
			scaled_data.append(
				self.scaler.transform(feats[i].T).T
			)

		feats_scaled_transposed = np.stack(scaled_data).transpose([0,2,1])

		self.model.fit(
			feats_scaled_transposed,
			labels,
			batch_size=self.batch_size,
			epochs=self.epochs,
			callbacks=self.callbacks,
			validation_split=self.validation_split,
			shuffle=True,
			verbose=self.fit_verbose,
			workers=self.n_workers)

		return self

	def predict(self, X):
		""" Predicts class of relationship between id1 and id2

		Predicts 1 or 0. 1 if id1 is predicted ranked higher, else 0

		This implementation should not actually be used, always predicts 1. 
		Purpose is only as function prototype.

		Args:
			

		Returns:
			array of length n_samples were each value is the predicted 
				relationship between the ids at that index
		"""
		if self.verbose > 0:
			print("\tScaling data")

		scaled_data = []
		for i in range(X.shape[0]):
			scaled_data.append(
				self.scaler.transform(X[i].T).T
			)

		feats_scaled_transposed = np.stack(scaled_data).transpose([0,2,1])

		if self.verbose > 0:
			print("\tPredicting")

		res = self.model.predict(feats_scaled_transposed,
								 batch_size=feats_scaled_transposed.shape[0],
								 verbose=self.fit_verbose,
								 workers=self.n_workers)
		
		res[res > .5] = 1
		res[res <= .5] = 0

		return res

	def predict_proba(self, X):
		""" Predicts probability id1 is ranked higher than id2

		Args:
			
		Returns:
			array of length n_samples were each value is the probability that
				the id1 at that index is ranked higher than the id2 at that 
				index
		"""
		if self.verbose > 0:
			print("\tScaling data")

		scaled_data = []
		for i in range(X.shape[0]):
			scaled_data.append(
				self.scaler.transform(X[i].T).T
			)

		feats_scaled_transposed = np.stack(scaled_data).transpose([0,2,1])

		if self.verbose > 0:
			print("\tPredicting")

		return self.model.predict(feats_scaled_transposed,
								  batch_size=feats_scaled_transposed.shape[0],
								  verbose=self.fit_verbose,
								  workers=self.n_workers)

		
class TimeSeriesComparerNoScaler(TimeSeriesComparer):
	"""
	Uses Keras model to compare time series. does not scale input
	"""

	def __init__(self, model, desc=None, 
				 batch_size=None, epochs=10, callbacks=None, validation_split=.1,
				 verbose=0, fit_verbose=1, n_workers=1):
		""" 
		verbose 3 or higher will save plots of model fitting history
		"""
		self.model = model
		self.desc = desc
		self.batch_size = batch_size
		self.epochs = epochs
		self.callbacks = callbacks
		self.verbose = verbose
		self.fit_verbose = fit_verbose
		self.validation_split = validation_split
		self.n_workers = n_workers

	def __str__(self):
		if self.desc is None:
			return "TSCns: {}".format(str(self.model))
		return "TSCns: {}".format(self.desc)

	def fit(self, feats, labels):
		"""
		
		Args:

		Returns:
			fit self
		"""

		self.model.fit(
			np.stack(feats).transpose([0,2,1]),
			labels,
			batch_size=self.batch_size,
			epochs=self.epochs,
			callbacks=self.callbacks,
			validation_split=self.validation_split,
			shuffle=True,
			verbose=self.fit_verbose,
			workers=self.n_workers)

		return self

	def predict(self, X):
		""" Predicts class of relationship between id1 and id2

		Predicts 1 or 0. 1 if id1 is predicted ranked higher, else 0

		This implementation should not actually be used, always predicts 1. 
		Purpose is only as function prototype.

		Args:
			

		Returns:
			array of length n_samples were each value is the predicted 
				relationship between the ids at that index
		"""

		if self.verbose > 0:
			print("\tPredicting")

		feats_transposed = np.stack(X).transpose([0,2,1])

		res = self.model.predict(feats_transposed,
								 batch_size=feats_transposed.shape[0],
								 verbose=self.fit_verbose,
								 workers=self.n_workers)
		
		res[res > .5] = 1
		res[res <= .5] = 0

		return res

	def predict_proba(self, X):
		""" Predicts probability id1 is ranked higher than id2

		Args:
			
		Returns:
			array of length n_samples were each value is the probability that
				the id1 at that index is ranked higher than the id2 at that 
				index
		"""

		if self.verbose > 0:
			print("\tPredicting")

		feats_transposed = np.stack(X).transpose([0,2,1])

		return self.model.predict(feats_transposed,
								  batch_size=feats_transposed.shape[0],
								  verbose=self.fit_verbose,
								  workers=self.n_workers)

