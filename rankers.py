import os
from collections import defaultdict
import pickle
import warnings

import numpy as np

import rbo  # https://github.com/changyaochen/rbo

from comparers import Comparer


def jaccard_similarity(list1, list2):
	s1 = set(list1)
	s2 = set(list2)
	return len(s1.intersection(s2)) / len(s1.union(s2))


def get_volume_n_days_before(events, survey_time, n_days):
    return len(events[(events > (survey_time - n_days * 86400))
                      & (events <= survey_time)])


def get_hawkes_signal(event_times, observation_time, beta):
	times_before_obs = event_times[event_times <= observation_time]
	time_deltas = observation_time - times_before_obs
	return np.sum(beta * np.exp(-beta * time_deltas))


def get_forget_intensity(lifetime, mu, theta, forget_type):
		if forget_type == 'pow':
			return np.log(mu / theta) / np.log(lifetime)
		elif forget_type == 'exp':
			return np.log(mu / theta) / lifetime


def forget_func(forget_type, time_delta, forget_intensity):
	if forget_type == 'pow':
		return max(1, time_delta) ** (-1 * forget_intensity)
	elif forget_type == 'exp':
		return np.e ** (-1 * forget_intensity * time_delta)


def get_cogsnet_signal(event_times, observation_time, mu, theta, forget_type, forget_intensity):

	current_signal = 0

	if len(event_times) > 0:
		while event_times[0] > observation_time:
			return current_signal

		current_signal = mu

	for i in range(1, len(event_times)):
		while event_times[i] > observation_time:
			val_at_obs = current_signal * forget_func(
                            forget_type,
                            (observation_time -
                             event_times[i-1]) / 3600,
                            forget_intensity)

			if val_at_obs < theta:
				return 0
			else:
				return val_at_obs

		decayed_signal = current_signal * forget_func(forget_type,
                                                (event_times[i] -
                                                 event_times[i-1]) / 3600,
                                                forget_intensity)
		if decayed_signal < theta:
			decayed_signal = 0
		current_signal = mu + decayed_signal * (1 - mu)

	# hits end of events without observation occuring yet
	val_at_obs = current_signal * forget_func(
				forget_type,
            	(observation_time - event_times[-1]) / 3600,
				forget_intensity)

	if val_at_obs < theta:
		return 0
	else:
		return val_at_obs


class Ranker():
	""" Superclass of Rankers, which are used to predict closeness based 
	on events.
	"""

	def __init__(self):
		pass

	def fit(self, interaction_dict=None, survey_dict=None):
		""" Fits Ranker to interaction and survey data """
		pass

	def _rank(self, data, top_n, survey_time=None):
		""" ranks the top_n possible ids in data """
		return np.asarray([*data.keys()])[:top_n]

	def predict(self, interaction_dict, survey_dict):
		""" For each survey in survey_dict, predicts a ranking of the top n
		closest nodes. This base Ranker does not predict the n first other ids
		respondant id interacted with.

		n is the length of others listed in the survey being predicted for.

		TODO: define arg dicts well

		Args:
			interaction_dict:
			survey_dict:

		Returns:
			nested dictionaries in the form of survey_dict, but innermost survey
				responses is an array that is the predicted ranking results in 
				order from lowest to highest rank instead of a dictionary with
				ranks like survey dict is
		"""
		pred_dict = defaultdict(lambda: defaultdict())

		for respondant_id, surveys in survey_dict.items():
			if (respondant_id not in interaction_dict.keys()):
				print("IDK if this should ever happen")
				continue

			for survey_time, survey in surveys.items():
				# this number of ids will be returned in ranking
				survey_n = len(survey)

				pred_dict[respondant_id][survey_time] = self._rank(
					interaction_dict[respondant_id], survey_n, survey_time)

		# make returned dict not a defualt dict
		for key in pred_dict.keys():
			pred_dict[key] = dict(pred_dict[key])
		pred_dict = dict(pred_dict)

		return pred_dict

	def score(self, interaction_dict, survey_dict):
		""" Predicts ranking using self.predict, then scores predictions and 
		returns a dictionary of scoring metrics.

		Args:
			interaction_dict:
			survey_dict:

		Returns:
			returns a dictionary where the keys are the names of scoring metrics
				and the values are the corresponding scores
		"""
		predicted_rankings = self.predict(interaction_dict, survey_dict)

		jaccards = []
		rbos = []

		for respondant_id, predictions in predicted_rankings.items():
			for survey_time, pred_ranking in predictions.items():
				survey_ranking = list(
					survey_dict[respondant_id][survey_time].values())

				jaccards.append(
					jaccard_similarity(survey_ranking, pred_ranking))
				rbos.append(
					rbo.RankingSimilarity(survey_ranking, pred_ranking).rbo())

		return {
			"jaccard": np.mean(jaccards),
			"rbo": np.mean(rbos)
		}


class RandomRanker(Ranker):
	""" Predicts by randomly selecting from the ids respondant has had events
	with before time of survey.
	"""

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		canidate_ids = [
			k for k, v in data.items() if np.any(v[:, 2] <= survey_time)]
		np.random.shuffle(canidate_ids)

		return canidate_ids[:top_n]


class VolumeRanker(Ranker):
	""" Ranks by number of events between two ids at time of survey. """

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		canidate_event_counts = {
                    k: len(v[np.asarray(v[:, 2] <= survey_time)])
                   	for k, v in data.items()
                   	if np.any(v[:, 2] <= survey_time)
                }

		ordered_inds = (
                    -np.asarray(list(canidate_event_counts.values()))).argsort()
		ordered_canidates = np.asarray(
                    list(canidate_event_counts.keys()))[ordered_inds]

		return ordered_canidates[:top_n]


class WindowedVolumeRanker(Ranker):
	""" Ranks by number of events between two ids in the time window before
	survey time. 
	"""

	def __init__(self, window_size=21):
		""" window_size is number of 24 hour periods before survey time that 
		time window includes.
		"""
		self.window_size = window_size

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		canidate_event_counts = {
			k: get_volume_n_days_before(v[:, 2], survey_time, self.window_size)
			for k, v in data.items()
			if np.any(v[:, 2] <= survey_time)
		}

		ordered_inds = (
                    -np.asarray(list(canidate_event_counts.values()))).argsort()
		ordered_canidates = np.asarray(
                    list(canidate_event_counts.keys()))[ordered_inds]

		return ordered_canidates[:top_n]


class RecencyRanker(Ranker):
	""" Ranks by most frequent event such the id with the most recent event to 
	the time of the survey is ranked first. 
	"""

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		most_recent_timestamp = {
			k: max(v[np.asarray(v[:, 2] <= survey_time)][:, 2])
			for k, v in data.items()
			if np.any(v[:, 2] <= survey_time)
		}

		ordered_inds = (
                    -np.asarray(list(most_recent_timestamp.values()))).argsort()
		ordered_canidates = np.asarray(
                    list(most_recent_timestamp.keys()))[ordered_inds]

		return ordered_canidates[:top_n]


class HawkesRanker(Ranker):
	""" Ranks using a hawkes process to predict signal strength. """

	def __init__(self, beta):
		""" set beta for hawkes process """
		self.beta = beta

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		hawkes_signals = {
			k: get_hawkes_signal(v[:, 2], survey_time, self.beta)
			for k, v in data.items()
			if np.any(v[:, 2] <= survey_time)
		}

		ordered_inds = (
                    -np.asarray(list(hawkes_signals.values()))).argsort()
		ordered_canidates = np.asarray(
                    list(hawkes_signals.keys()))[ordered_inds]

		return ordered_canidates[:top_n]


class PresetParamCogSNetRanker(Ranker):
	""" Ranks using CogSNet. Requires parameters are already known, therefore
	fit does nothing with this class with regards to actually fitting the data.

	Use is for testing one set of params, either on their own or as part 
	of CogSNetRanker's parallelized grid search
	"""

	def __init__(self, L, mu, theta, forget_type):
		""" L is number of days """
		self.L = L * 24
		self.mu = mu
		self.theta = theta
		self.forget_type = forget_type
		self.forget_intensity = get_forget_intensity(self.L, mu, theta, forget_type)

	def fit(self, interaction_dict=None, survey_dict=None):
		""" Fits Ranker to interaction and survey data """
		warnings.warn("Fitting a PresetParamCogSNetRanker has no effect.")

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		cogsnet_signals = {
					k: get_cogsnet_signal(v[:, 2], survey_time, self.mu, 
											self.theta, self.forget_type,
											self.forget_intensity)
						for k, v in data.items()
				}

		ordered_inds = (
                    -np.asarray(list(cogsnet_signals.values()))).argsort()
		ordered_canidates = np.asarray(
                    list(cogsnet_signals.keys()))[ordered_inds]

		return ordered_canidates[:top_n]

class PairwiseRanker(Ranker):
	""" Ranks using a Comparer 

	Given an integer n ≥ 2, we consider a collection of n items, indexed by the 
	set [n] : = {1, . . . , n}. For each pair i != j, we let Mij denote the 
	probability that item i wins the comparison with item j. We assume that each
	comparison necessarily results in one winner, meaning that:
	
		Mij + Mji = 1, and Mii = 1/2

	where we set the diagonal as 1/2 for concreteness.

	For any item i ∈ [n], we define an associated score τi as:

		τi(M) := (1/n) Σ(j = 1 to n) Mij 

	In words, the score τi(M) of any item i ∈ [n] corresponds to the probability 
	that item i beats an item chosen uniformly at random from all n items. 

	(above taken from http://jmlr.org/papers/volume18/16-206/16-206.pdf)

	For the purpose of ranking, we will use τi as the score of any item i.
	"""

	def __init__(self, comparer):
		""" comparer is a Comparer or subclass (see comparers.py) """
		self.comparer = comparer

	def _create_indiv_feat_vec(self, indiv_data, survey_time=None):
		""" Create the n x 1 feature vector to represent an individual at the
		time of the survey in the pairwise comparison
		"""
		return indiv_data[:, 2]

	def _rank(self, data, top_n, survey_time):
		""" Create M matrix and then rank based on taus """
		global d 
		d = data 

		global s 
		s = survey_time

		global i
		global m

		node_ids = data.keys()
		id_to_ind = {n_id: ind for ind, n_id in enumerate(node_ids)}
		id_to_feats = {n_id: self._create_indiv_feat_vec(data[n_id], survey_time) 
						for n_id in node_ids}

		M = np.eye(len(node_ids), len(node_ids)) * .5

		for i_id in node_ids:
			for j_id in node_ids:
				# diagonal already filled with .5 
				if i_id == j_id:
					continue

				M[id_to_ind[i_id], id_to_ind[j_id]] = self.comparer.predict_proba(
					id_to_feats[i_id],
					id_to_feats[j_id]
				)

		i = id_to_feats
		m = M

		# cogsnet_signals = {
        #             k: get_cogsnet_signal(v[:, 2], survey_time, self.mu,
        #                                   self.theta, self.forget_type,
        #                                   self.forget_intensity)
        #             for k, v in data.items()
        #         }

		# ordered_inds = (
        #             -np.asarray(list(cogsnet_signals.values()))).argsort()
		# ordered_canidates = np.asarray(
        #             list(cogsnet_signals.keys()))[ordered_inds]

		# return ordered_canidates[:top_n]

	
	

if __name__ == "__main__":
	""" main is used to test Ranker classes """

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)
	
	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	# Ranker
	ranker = Ranker()

	ranker.fit(interaction_dict, survey_dict)
	ranker.fit()

	rp = ranker.predict(interaction_dict, survey_dict)
	rs = ranker.score(interaction_dict, survey_dict)

	# # RandomRanker
	# rand_ranker = RandomRanker()

	# rand_ranker.fit(interaction_dict, survey_dict)
	# rand_ranker.fit()

	# rrp = rand_ranker.predict(interaction_dict, survey_dict)
	# rrs = rand_ranker.score(interaction_dict, survey_dict)

	# # VolumeRanker
	# vol_ranker = VolumeRanker()

	# vol_ranker.fit(interaction_dict, survey_dict)
	# vol_ranker.fit()

	# vrp = vol_ranker.predict(interaction_dict, survey_dict)
	# vrs = vol_ranker.score(interaction_dict, survey_dict)

	# # WindowedVolumeRanker
	# win_vol_ranker_21 = WindowedVolumeRanker()
	# win_vol_ranker_7 = WindowedVolumeRanker(7)

	# win_vol_ranker_21.fit(interaction_dict, survey_dict)
	# win_vol_ranker_7.fit()

	# vrp21 = win_vol_ranker_21.predict(interaction_dict, survey_dict)
	# vrs21 = win_vol_ranker_21.score(interaction_dict, survey_dict)

	# vrp7 = win_vol_ranker_7.predict(interaction_dict, survey_dict)
	# vrs7 = win_vol_ranker_7.score(interaction_dict, survey_dict)

	# # RecencyRanker
	# rec_ranker = RecencyRanker()

	# rec_ranker.fit(interaction_dict, survey_dict)
	# rec_ranker.fit()

	# recp = rec_ranker.predict(interaction_dict, survey_dict)
	# recs = rec_ranker.score(interaction_dict, survey_dict)

	# # HawkesRanker
	# hawkes_ranker = HawkesRanker(1.727784e-07)
	# hawkes_ranker_2 = HawkesRanker(beta=.00001)

	# hawkes_ranker.fit(interaction_dict, survey_dict)
	# hawkes_ranker_2.fit()

	# hrp = hawkes_ranker.predict(interaction_dict, survey_dict)
	# hrs = hawkes_ranker.score(interaction_dict, survey_dict)

	# hrp2 = hawkes_ranker_2.predict(interaction_dict, survey_dict)
	# hrs2 = hawkes_ranker_2.score(interaction_dict, survey_dict)

	# # CogSNetRanker
	# cogsnet_ranker = PresetParamCogSNetRanker(L=21.0, mu=0.018915, theta=0.017932,
	# 							forget_type='exp')

	# cogsnet_ranker.fit(interaction_dict, survey_dict)
	# cogsnet_ranker.fit()

	# cnp = cogsnet_ranker.predict(interaction_dict, survey_dict)
	# cns = cogsnet_ranker.score(interaction_dict, survey_dict)

	# PairwiseRanker
	pw_ranker = PairwiseRanker(Comparer())

	pw_ranker.fit(interaction_dict, survey_dict)
	pw_ranker.fit()

	cnp = pw_ranker.predict(interaction_dict, survey_dict)
	# cns = pw_ranker.score(interaction_dict, survey_dict)
