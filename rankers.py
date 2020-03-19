import os
from collections import defaultdict
import pickle
import warnings

import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences 

import rbo  # https://github.com/changyaochen/rbo

from rankers_util import *


class Ranker():
	""" Superclass of Rankers, which are used to predict closeness based 
	on events.
	"""

	def __init__(self):
		pass

	def __str__(self):
		return "Ranker"

	def fit(self, interaction_dict=None, survey_dict=None):
		""" Fits Ranker to interaction and survey data """
		return self

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
			if respondant_id not in interaction_dict.keys():
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
		kendal_taus = []

		for respondant_id, predictions in predicted_rankings.items():
			for survey_time, pred_ranking in predictions.items():
				survey_ranking = list(
					survey_dict[respondant_id][survey_time].values())

				jaccards.append(
					jaccard_similarity(survey_ranking, pred_ranking))

				rbos.append(
					rbo.RankingSimilarity(survey_ranking, pred_ranking).rbo())

				kendal_taus.append(kendal_tau(survey_ranking, pred_ranking))

		kendal_taus = np.asarray(kendal_taus)
		kendal_taus = kendal_taus[kendal_taus != np.array(None)]

		return {
			"jaccard": np.mean(jaccards),
			"rbo": np.mean(rbos),
			"kendall_tau": np.mean(kendal_taus)
		}

	def predict_and_score(self, interaction_dict, survey_dict):
		""" 
		Predicts ranking using self.predict, then scores in same way as
		self.score. Returns result of both while only ranking once.

		Args:
			interaction_dict:
			survey_dict:

		Returns:
			returns a dictionary with two keys. One key is 'predict' whose 
				value is the result of self.predict. The other key is 
				'score' whose value is the result of self.predict
		"""
		predicted_rankings = self.predict(interaction_dict, survey_dict)

		jaccards = []
		rbos = []
		kendal_taus = []

		for respondant_id, predictions in predicted_rankings.items():
			for survey_time, pred_ranking in predictions.items():
				survey_ranking = list(
					survey_dict[respondant_id][survey_time].values())

				jaccards.append(
					jaccard_similarity(survey_ranking, pred_ranking))

				rbos.append(
					rbo.RankingSimilarity(survey_ranking, pred_ranking).rbo())

				kendal_taus.append(kendal_tau(survey_ranking, pred_ranking))

		kendal_taus = np.asarray(kendal_taus)
		kendal_taus = kendal_taus[kendal_taus != np.array(None)]

		return {
			'predict': predicted_rankings,
			'score':	{"jaccard": np.mean(jaccards),
							"rbo": np.mean(rbos),
							"kendall_tau": np.mean(kendal_taus)}
		}

	def get_signals(self, ids, interaction_dict):
		raise NotImplementedError("Cannot get signals with this ranker type")


class RandomRanker(Ranker):
	""" Predicts by randomly selecting from the ids respondant has had events
	with before time of survey.
	"""

	def __str__(self):
		return "RandomRanker"

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		canidate_ids = [
			k for k, v in data.items() if np.any(v[:, 2] <= survey_time)]
		np.random.shuffle(canidate_ids)

		return canidate_ids[:top_n]


class VolumeRanker(Ranker):
	""" Ranks by number of events between two ids at time of survey. """

	def __str__(self):
		return "VolumeRanker"

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

	def get_signals(self, ids, users_interaction_dict, times, norm_all=True):
		""" 
		Gives signal at given times
		
		signal is (n events) / (max n events for all edges) 

		Args:
			users_interaction_dict is the interactions for just the users whose
				signals are being generated

			norm_all: If Ture normalizes with all edge all time min and
				max. If False normalizes with min and max per time in times
		"""
		counts = {
			k: [len(v[np.asarray(v[:, 2] <= times[0])])]
			for k, v in users_interaction_dict.items()
		}

		for time in times[1:]:
			for k, v in users_interaction_dict.items():
				counts[k].append(len(v[np.asarray(v[:, 2] <= time)]))
		
		if norm_all:
			max_events = np.max(np.hstack(list(counts.values())))
		else:
			max_events = np.max(np.stack(list(counts.values())), axis=0)
			max_events[max_events == 0] = 1 

		ret_dict = dict()

		for uid in ids:
			if uid in counts.keys():
				ret_dict[uid] = np.asarray(counts[uid]) / max_events
			else:
				ret_dict[uid] = np.zeros(len(times))

		return ret_dict

class WindowedVolumeRanker(Ranker):
	""" Ranks by number of events between two ids in the time window before
	survey time. 
	"""

	def __init__(self, window_size=21):
		""" window_size is number of 24 hour periods before survey time that 
		time window includes.
		"""
		self.window_size = window_size

	def __str__(self):
		return "WindowRanker ws={}".format(self.window_size)

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

	def get_signals(self, ids, users_interaction_dict, times, norm_all=True):
		""" 
		Gives signal at given times
		
		signal = (n events in window) / (max n events in any window for all edges)
		"""
		counts = {
			k: [get_volume_n_days_before(v[:, 2], times[0], self.window_size)]
			for k, v in users_interaction_dict.items()
		}
		
		for time in times[1:]:
			for k, v in users_interaction_dict.items():
				counts[k].append(
					get_volume_n_days_before(v[:, 2], time, self.window_size)
				)
		
		if norm_all:
			max_events = np.max(np.hstack(list(counts.values())))
		else:
			max_events = np.max(np.stack(list(counts.values())), axis=0)
			max_events[max_events == 0] = 1 

		ret_dict = dict()

		for uid in ids:
			if uid in counts.keys():
				ret_dict[uid] = np.asarray(counts[uid]) / max_events
			else:
				ret_dict[uid] = np.zeros(len(times))

		return ret_dict


class RecencyRanker(Ranker):
	""" Ranks by most frequent event such the id with the most recent event to 
	the time of the survey is ranked first. 
	"""

	def __str__(self):
		return "RecencyRanker"

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

	def __str__(self):
		return "HawkesRanker beta={}".format(self.beta)

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		hawkes_signals = {
			k: get_hawkes_signal(v[:, 2], survey_time, self.beta)
			for k, v in data.items()
			if np.any(v[:, 2] <= survey_time)
		}

		ordered_inds = (-np.asarray(list(hawkes_signals.values()))).argsort()
		ordered_canidates = np.asarray(
                    list(hawkes_signals.keys()))[ordered_inds]

		return ordered_canidates[:top_n]

	def get_signals(self, ids, users_interaction_dict, times):
		""" Min max normalized hawkes process value """
		hawkes_signals = {
			k: [get_hawkes_signal(v[:, 2], times[0], self.beta)]
			for k, v in users_interaction_dict.items()
		}
		
		for time in times[1:]:
			for k, v in users_interaction_dict.items():
				hawkes_signals[k].append(
					get_hawkes_signal(v[:, 2], time, self.beta)
				)
		
		max_signal = np.max(np.hstack(list(hawkes_signals.values())))

		ret_dict = dict()

		for uid in ids:
			if uid in hawkes_signals.keys():
				ret_dict[uid] = np.asarray(hawkes_signals[uid]) / max_signal
			else:
				ret_dict[uid] = np.zeros(len(times))

		return ret_dict


class HawkesRankerL(HawkesRanker):
	""" 
	Ranks using a hawkes process to predict signal strength. 
	
	Takes args of L and p to set beta
	"""

	def __init__(self, L, p):
		""" set beta for hawkes process """
		self.L = L 
		self.p = p
		self.beta = np.log(1/p) / (L * 3600 * 24)

	def __str__(self):
		return "HawkesRanker L={} p={} beta={}".format(self.L, self.p, self.beta)

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


class EnHawkesRanker(Ranker):
	""" 
	enhanced hawkes, but its actualy not so don't call it that
	"""

	def __init__(self, L1, p, L2):
		""" set beta for hawkes process """
		self.L1 = L1
		self.p = p
		self.L2 = L2
		self.beta = np.log(1/p) / (L1 * 3600 * 24)

	def __str__(self):
		return "EnHawkesRanker L1={} p={} L2={} beta={}".format(
			self.L1, self.p, self.L2, self.beta)

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		hawkes_signals = {
			k: get_en_hawkes_signal(v[:, 2], survey_time, self.beta, self.L2)
			for k, v in data.items()
			if np.any(v[:, 2] <= survey_time)
		}

		ordered_inds = (
                    -np.asarray(list(hawkes_signals.values()))).argsort()
		ordered_canidates = np.asarray(
                    list(hawkes_signals.keys()))[ordered_inds]

		return ordered_canidates[:top_n]


class CogSNetRanker(Ranker):
	""" Ranks using CogSNet. """

	def __init__(self, L, mu, theta, forget_type='exp', desc_str=None):
		""" L is number of days """
		self.L = L * 24
		self.mu = mu
		self.theta = theta
		self.forget_type = forget_type
		self.forget_intensity = get_forget_intensity(self.L, mu, theta, forget_type)
		self.desc_str = desc_str

	def __str__(self):
		if self.desc_str is None:
			return "CogSNetRanker L={} mu={} theta={} f_type={}".format(
				self.L, self.mu, self.theta, self.forget_type
			)
		else:
			return self.desc_str

	def fit(self, interaction_dict=None, survey_dict=None):
		""" Fits Ranker to interaction and survey data """
		warnings.warn("Fitting a PresetParamCogSNetRanker has no effect.")
		return self

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

	def get_signals(self, ids, users_interaction_dict, times):
		""" CogSNet values """
		cogsnet_signals = {
			k: [get_cogsnet_signal(v[:, 2], times[0], self.mu, 
									self.theta, self.forget_type,
									self.forget_intensity)]
				for k, v in users_interaction_dict.items()
		}
		
		for time in times[1:]:
			for k, v in users_interaction_dict.items():
				cogsnet_signals[k].append(
					get_cogsnet_signal(v[:, 2], time, self.mu, 
										self.theta, self.forget_type,
										self.forget_intensity)
				)
		
		max_signal = np.max(np.hstack(list(cogsnet_signals.values())))

		ret_dict = dict()

		for uid in ids:
			if uid in cogsnet_signals.keys():
				ret_dict[uid] = np.asarray(cogsnet_signals[uid])
			else:
				ret_dict[uid] = np.zeros(len(times))

		return ret_dict


class CogSNet2Ranker(Ranker):
	""" Ranks using CogSNet v2. """

	def __init__(self, L, mu, desc_str=None):
		""" L is number of days """
		self.L = L * 24
		self.mu = mu
		self.desc_str = desc_str

	def __str__(self):
		if self.desc_str is None:
			return "CogSNet2Ranker L={} mu={}".format(
				self.L, self.mu
			)
		else:
			return self.desc_str

	def fit(self, interaction_dict=None, survey_dict=None):
		""" Fits Ranker to interaction and survey data """
		warnings.warn("Fitting a PresetParamCogSNetRanker has no effect.")
		return self

	def _forget_func(self, time_delta):
		if time_delta < self.L:
			return (1 / np.log( self.L + 1)) * np.log(-time_delta + self.L + 1)  
		
		return 0

	def _get_c2_signals(self, event_times, observation_time):
		current_signal = 0

		if len(event_times) > 0:
			while event_times[0] > observation_time:
				return current_signal

			current_signal = self.mu

		for i in range(1, len(event_times)):
			while event_times[i] > observation_time:
				val_at_obs = current_signal * self._forget_func(
					(observation_time - event_times[i-1]) / 3600)

				return val_at_obs

			decayed_signal = current_signal * self._forget_func(
					(observation_time - event_times[i-1]) / 3600)
					
			current_signal = self.mu + decayed_signal * (1 - self.mu)

		# hits end of events without observation occuring yet
		val_at_obs = current_signal * self._forget_func(
					(observation_time - event_times[-1]) / 3600)

		return val_at_obs

	def _rank(self, data, top_n, survey_time):
		""" ranks the top_n possible ids in data """
		cogsnet_signals = {
			k: self._get_c2_signals(v[:, 2], survey_time)
				for k, v in data.items()
		}

		ordered_inds = (
                    -np.asarray(list(cogsnet_signals.values()))).argsort()
		ordered_canidates = np.asarray(
                    list(cogsnet_signals.keys()))[ordered_inds]

		return ordered_canidates[:top_n]

	def get_signals(self, ids, users_interaction_dict, times):
		""" CogSNet values """
		cogsnet_signals = {
			k: [self._get_c2_signals(v[:, 2], [times[0]])]
				for k, v in users_interaction_dict.items()
		}
		
		for time in times[1:]:
			for k, v in users_interaction_dict.items():
				cogsnet_signals[k].append(
					self._get_c2_signals(v[:, 2], time)
				)
		
		ret_dict = dict()

		for uid in ids:
			if uid in cogsnet_signals.keys():
				ret_dict[uid] = np.asarray(cogsnet_signals[uid])
			else:
				ret_dict[uid] = np.zeros(len(times))

		return ret_dict


class PairwiseRanker(Ranker):
	""" 
	Ranks using a Comparer, cannot use TimeSeriesComparers

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

	For the purpose of ranking, we will use τi as the score of any item i if tau
	ranking_method is used. Otherwise use borda count method.
	"""

	def __init__(self, comparer, rank_method='tau', desc_str=None, verbose=0):
		""" 
		comparer is a Comparer or subclass (see comparers.py) 
		
		Args:
			comparer: Comparer object
			rank_method: 'tau' or 'borda'
			verbose: prints updates while fitting if > 0
		"""
		self.comparer = comparer
		self.verbose = verbose
		self.rank_method = rank_method
		self.desc_str = desc_str

	def __str__(self):
		if self.desc_str is None:
			return "PairwiseRanker {}: {}".format(
				self.rank_method, str(self.comparer))
		else:
			return self.desc_str

	def fit(self, interaction_dict, survey_dict):
		""" 
		fits underlying comparer using all possible pairwise relationships
		in survey dict data
		"""
		X_1 = []
		X_2 = []
		y = []

		if self.verbose > 0:
			print("\tGenerating data to train model on")

		for respondant_id, surveys in survey_dict.items():
			if respondant_id not in interaction_dict.keys():
				continue

			for survey_time, survey in surveys.items():
				id_to_rank = {n_id: rank for rank, n_id in survey.items()}
				
				all_ids = list(interaction_dict[respondant_id].keys())
				num_ids = len(all_ids)
				feat_vecs = [self._create_indiv_feat_vec(
									interaction_dict[respondant_id][n_id], 
									survey_time)
								for n_id in all_ids]

				ranks = np.full(num_ids, num_ids)

				for i in range(num_ids):
					if all_ids[i] in id_to_rank.keys():
						ranks[i] = id_to_rank[all_ids[i]]

				for i in range(num_ids):
					for j in range(num_ids):
						# dont need to check if i == j b/c will have equal rank
						if ranks[i] < ranks[j]:
							X_1.append(feat_vecs[i])
							X_2.append(feat_vecs[j])
							y.append(1)
						elif ranks[i] > ranks[j]:
							X_1.append(feat_vecs[i])
							X_2.append(feat_vecs[j])
							y.append(0)

		if self.verbose > 0:
			print("\tFitting comparer")
		self.comparer.fit(X_1, X_2, y)
		return self


	def _create_indiv_feat_vec(self, indiv_data, survey_time):
		""" 
		Create the n x 1 feature vector to represent an individual at the
		time of the survey in the pairwise comparison

		Features (in order they are in resulting array):
			volume: number of events with id at survey time
			21 day window volume: number of events with id in window before 
				survey time
			7 day window volume: number of events with id in window before 
				survey time
			recency: time between survey time and most recent event
			cogsnet signal: cogsnet signal at survey time using given params
			hawkes process signal: at survey time using given params
		"""
		events =  indiv_data[:, 2]

		L = 21
		mu = 0.2
		theta = 0.166667
		forget_type = 'exp'
		forget_intensity = get_forget_intensity(L, mu, theta, forget_type)

		feat_vec = [
			get_volume(events, survey_time),
			get_volume_n_days_before(events, survey_time, 21),
			get_volume_n_days_before(events, survey_time, 152),
			get_recency(events, survey_time),
			get_cogsnet_signal(events, survey_time, mu, theta, forget_type, 
								forget_intensity),
			get_hawkes_signal(events, survey_time, beta=1.727784e-07)
		]

		return np.asarray(feat_vec)

	def _rank(self, data, top_n, survey_time):
		if self.rank_method == 'tau':
			return self._rank_tau(data, top_n, survey_time)
		if self.rank_method == 'borda':
			return self._rank_borda(data, top_n, survey_time)

		raise ValueError("PairwiseRanker must have rank_method be 'tau' or 'borda'")

	def _rank_tau(self, data, top_n, survey_time):
		""" Create M matrix and then rank based on taus """

		node_ids = list(data.keys())
		id_to_feats = {n_id: self._create_indiv_feat_vec(data[n_id], survey_time) 
						for n_id in node_ids}

		X_1 = []
		X_2 = []

		for i_id in node_ids:
			for j_id in node_ids:
				X_1.append(id_to_feats[i_id])
				X_2.append(id_to_feats[j_id])
				
		M = self.comparer.predict_proba(X_1, X_2)
		M = M.reshape((len(node_ids), len(node_ids)))
		np.fill_diagonal(M, .5)

		taus = np.sum(M, axis=1) / len(node_ids)

		ordered_inds = (-taus).argsort()
		ordered_canidates = np.asarray(node_ids)[ordered_inds]

		return ordered_canidates[:top_n]

	def _rank_borda(self, data, top_n, survey_time):
		""" Create M matrix and then rank based on borda count """

		node_ids = list(data.keys())
		id_to_feats = {n_id: self._create_indiv_feat_vec(data[n_id], survey_time)
                 for n_id in node_ids}

		X_1 = []
		X_2 = []

		for i_id in node_ids:
			for j_id in node_ids:
				X_1.append(id_to_feats[i_id])
				X_2.append(id_to_feats[j_id])

		M = self.comparer.predict(X_1, X_2)
		M = M.reshape((len(node_ids), len(node_ids)))
		M[M == 0] = -1
		np.fill_diagonal(M, 0)

		counts = np.sum(M, axis=1)

		ordered_inds = (-counts).argsort()
		ordered_canidates = np.asarray(node_ids)[ordered_inds]

		return ordered_canidates[:top_n]
	
	def get_signals(self, ids, users_interaction_dict, times):
		if self.rank_method == 'tau':
			return self._get_signals_tau(ids, users_interaction_dict, times)
		if self.rank_method == 'borda':
			return self._get_signals_borda(ids, users_interaction_dict, times)

		raise ValueError("PairwiseRanker must have rank_method be 'tau' or 'borda'")

	def _get_signals_tau(self, ids, users_interaction_dict, times):
		node_ids = list(users_interaction_dict.keys())
		signals = {uid: [] for uid in node_ids}

		for time in times:
			id_to_feats = {
				n_id: self._create_indiv_feat_vec(users_interaction_dict[n_id], time)
					for n_id in node_ids
			}

			X_1 = []
			X_2 = []

			for i_id in node_ids:
				for j_id in node_ids:
					X_1.append(id_to_feats[i_id])
					X_2.append(id_to_feats[j_id])
					
			M = self.comparer.predict_proba(X_1, X_2)
			M = M.reshape((len(node_ids), len(node_ids)))
			np.fill_diagonal(M, .5)

			taus = np.sum(M, axis=1) / len(node_ids)

			for uid, tau in zip(node_ids, taus):
				signals[uid].append(tau)

		for uid in ids:
			if uid not in signals.keys():
				signals[uid] = np.zeros(len(times))

		return signals

	def _get_signals_borda(self, ids, users_interaction_dict, times):
		node_ids = list(users_interaction_dict.keys())
		borda_counts = {uid: [] for uid in node_ids}

		for time in times:
			id_to_feats = {n_id: self._create_indiv_feat_vec(
								users_interaction_dict[n_id], time
							)		
							for n_id in node_ids}

			X_1 = []
			X_2 = []

			for i_id in node_ids:
				for j_id in node_ids:
					X_1.append(id_to_feats[i_id])
					X_2.append(id_to_feats[j_id])

			M = self.comparer.predict(X_1, X_2)
			M = M.reshape((len(node_ids), len(node_ids)))
			M[M == 0] = -1
			np.fill_diagonal(M, 0)

			counts = np.sum(M, axis=1)

			for uid, count in zip(node_ids, counts):
				borda_counts[uid].append(count)

		ret_dict = dict()

		for uid in ids:
			if uid in borda_counts.keys():
				ret_dict[uid] = (np.asarray(borda_counts[uid]) 
									+ len(node_ids)) / (2 * len(node_ids))
			else:
				ret_dict[uid] = np.zeros(len(times))

		return ret_dict

class TimeSeriesPairwiseRanker(PairwiseRanker):
	""" 
	Ranks using TimeSeriesComparers

	Data used will be longest possible time series
	"""

	def __init__(self, comparer, rank_method='tau', desc_str=None, verbose=0,
					bin_size=1, window=None, other_feat=False, 
					text_call_split=False, metric='count',
					earliest_timestamp=1312617635):
		""" 
		comparer is a TimeSeriesComparer or subclass (see comparers.py) 
		
		Args:
			comparer: Comparer object
			rank_method: 'tau' or 'borda'
			desc_str: alternate string decription of model
			verbose: prints updates while fitting if > 0
			bin_size: bin size in n days (24 hour periods divide backward 
				from survey time)
			window: if None time series contains all avaibable data before
				survey, if number time series is this number of bins before
				survey time
			other_feat: generates a time series for combination of all people
				not being compared
			text_call_split: if True, generates one time series for call data
				and another for text. If False consolidated as one
			metric: 'count': time series based on count of calls and texts
					'val': time series based on column 1 value, only use when
							call and text are split
					'both': both count and val features, only use when split
			earliest_timestamp: for NetSense: 1312617635
								for reality commons: 1189000931
		"""
		self.comparer = comparer
		self.rank_method = rank_method
		self.desc_str = desc_str
		self.verbose = verbose
		self.bin_size = bin_size
		self.window = window
		self.other_feat = other_feat
		self.text_call_split = text_call_split
		self.metric = metric
		self.earliest_timestamp = earliest_timestamp

	def __str__(self):
		if self.desc_str is None:
			return "TimeSeriesPairwiseRanker {} bs_{} w_{} of_{} s_{} m_{}: {}".format(
				self.rank_method, 
				self.bin_size, 
				self.window,
				self.other_feat,
				self.text_call_split,
				self.metric,
				str(self.comparer))
		else:
			return self.desc_str

	def fit(self, interaction_dict, survey_dict):
		""" Generated data to fit with and then fits TimeSeriesComparer """

		if self.verbose > 0:
			print("\tGenerating data")
		X, y = self._generate_data(interaction_dict, survey_dict)

		if self.verbose > 0:
			print("\tFitting comparer")
		self.comparer.fit(X, y)

		return self

	def _create_indiv_time_series(self, indiv_data, survey_time):
		"""
		Creates time series for individual at survey time with Rankers
		params
		"""
		bin_width = self.bin_size * 3600 * 24

		if self.window is not None:
			earliest = survey_time - (bin_width * self.window)
		else:
			earliest = self.earliest_timestamp

		bins = [survey_time]
		next_time = survey_time - bin_width

		while next_time > earliest:
			bins.append(next_time)
			next_time = next_time - bin_width

		bins.append(next_time)
		bins.reverse()

		series = []

		# to_process holds (all data as one matirx) or (call data and 
		# 	text data as seperate matircs)

		if self.text_call_split:
			to_process = [
				indiv_data[indiv_data[:, 0] == 1], 	# text
				indiv_data[indiv_data[:, 0] == 0]	# call
			]
		else:
			to_process = [indiv_data]

		for data in to_process:
			bin_edge_inds = np.searchsorted(data[:,2], bins, side='right')

			if self.metric == 'count' or self.metric == 'both':
				counts = []
				for i in range(len(bin_edge_inds) - 1):
					counts.append(bin_edge_inds[i + 1] - bin_edge_inds[i])

				series.append(counts)

			if self.metric == 'val' or self.metric == 'both':
				totals = []
				for i in range(len(bin_edge_inds) - 1):
					totals.append(np.sum(
						data[:,1][bin_edge_inds[i] : bin_edge_inds[i + 1]]
					))

				series.append(totals)

		return np.vstack(series)

	def _generate_data(self, interaction_dict, survey_dict):
		""" 
		Generates time series feature vectors and labels for two people 
		being compared 
		"""

		X = []
		y = []

		if self.verbose > 0:
			n_ids = len(survey_dict.keys())
			n = 1

		for respondant_id, surveys in survey_dict.items():

			if self.verbose > 0:
				print("\t\t{}/{}".format(n, n_ids))
				n+=1

			if respondant_id not in interaction_dict.keys():
				continue

			for survey_time, survey in surveys.items():
				id_to_rank = {n_id: rank for rank, n_id in survey.items()}
				
				all_ids = list(interaction_dict[respondant_id].keys())
				num_ids = len(all_ids)
				time_series = np.stack(
					[self._create_indiv_time_series(
							interaction_dict[respondant_id][n_id], 
							survey_time)
						for n_id in all_ids])

				ranks = np.full(num_ids, num_ids)

				for i in range(num_ids):
					if all_ids[i] in id_to_rank.keys():
						ranks[i] = id_to_rank[all_ids[i]]

				if self.other_feat:
					series_sums = np.sum(time_series, axis=0)

					for i in range(num_ids):
						for j in range(num_ids):
							# dont need to check if i == j b/c will have equal rank
							if ranks[i] < ranks[j]:
								X.append(np.vstack(
									[time_series[i], time_series[j],
									 series_sums - time_series[i] - time_series[j]]
								))
								y.append(1)
							elif ranks[i] > ranks[j]:
								X.append(np.vstack(
									[time_series[i], time_series[j],
									 series_sums - time_series[i] - time_series[j]]
								))
								y.append(0)

				else:
					for i in range(num_ids):
						for j in range(num_ids):
							if ranks[i] < ranks[j]:
								X.append(np.vstack(
									[time_series[i], time_series[j]]
								))
								y.append(1)
							elif ranks[i] > ranks[j]:
								X.append(np.vstack(
									[time_series[i], time_series[j]]
								))
								y.append(0)

		max_ts_len = max((tsm.shape[1] for tsm in X))
		prepadded_X = [pad_sequences(tsm, maxlen=max_ts_len) for tsm in X]

		return np.stack(prepadded_X), np.asarray(y)

	def _generate_data_for_rank(self, interaction_dict, id_pairs, survey_time):
		""" interaction dict is just interaction fro person being ranked for.
		see _rank
		"""
		X = []

		if self.verbose > 0:
			print("\tGenerating data to rank")

		all_ids = list(interaction_dict.keys())
		id_to_time_series = {
			n_id: self._create_indiv_time_series(interaction_dict[n_id],
												 survey_time)
				for n_id in all_ids
		}

		time_series = np.stack(list(id_to_time_series.values()))

		if self.other_feat:
			series_sums = np.sum(time_series, axis=0)

			for id1, id2 in id_pairs:
				X.append(np.vstack([
					id_to_time_series[id1],
					id_to_time_series[id2],
					series_sums - id_to_time_series[id1] - id_to_time_series[id2]
				]))

		else:
			for id1, id2 in id_pairs:
				X.append(np.vstack([
					id_to_time_series[id1],
					id_to_time_series[id2]
				]))

		return np.stack(X)
		
	def _rank_tau(self, data, top_n, survey_time):
		""" Create M matrix and then rank based on taus """

		node_ids = list(data.keys())

		id_pairs = []
		for i_id in node_ids:
			for j_id in node_ids:
				id_pairs.append((i_id, j_id))

		if self.verbose > 0:
			print("Gen data in _rank_tau")

		X = self._generate_data_for_rank(data, id_pairs, survey_time)
				
		M = self.comparer.predict_proba(X)
		M = M.reshape((len(node_ids), len(node_ids)))
		np.fill_diagonal(M, .5)

		taus = np.sum(M, axis=1) / len(node_ids)

		ordered_inds = (-taus).argsort()
		ordered_canidates = np.asarray(node_ids)[ordered_inds]

		return ordered_canidates[:top_n]

	def _rank_borda(self, data, top_n, survey_time):
		""" Create M matrix and then rank based on borda count """

		node_ids = list(data.keys())

		id_pairs = []
		for i_id in node_ids:
			for j_id in node_ids:
				id_pairs.append((i_id, j_id))

		if self.verbose > 0:
			print("Gen data in _rank_borda")

		X = self._generate_data_for_rank(data, id_pairs, survey_time)

		M = self.comparer.predict(X)
		M = M.reshape((len(node_ids), len(node_ids)))
		M[M == 0] = -1
		np.fill_diagonal(M, 0)

		counts = np.sum(M, axis=1)

		ordered_inds = (-counts).argsort()
		ordered_canidates = np.asarray(node_ids)[ordered_inds]

		return ordered_canidates[:top_n]

	def _get_signals_tau(self, ids, users_interaction_dict, times):
		node_ids = list(users_interaction_dict.keys())
		signals = {uid: [] for uid in node_ids}

		for time in times:
			id_pairs = []
			for i_id in node_ids:
				for j_id in node_ids:
					id_pairs.append((i_id, j_id))

			if self.verbose > 0:
				print("Gen data for time in _get_signals_tau")

			X = self._generate_data_for_rank(
				users_interaction_dict,  
				id_pairs, 
				time)
					
			M = self.comparer.predict_proba(X)
			M = M.reshape((len(node_ids), len(node_ids)))
			np.fill_diagonal(M, .5)

			taus = np.sum(M, axis=1) / len(node_ids)

			for uid, tau in zip(node_ids, taus):
				signals[uid].append(tau)

		for uid in ids:
			if uid not in signals.keys():
				signals[uid] = np.zeros(len(times))

		return signals

	def _get_signals_borda(self, ids, users_interaction_dict, times):
		node_ids = list(users_interaction_dict.keys())
		borda_counts = {uid: [] for uid in node_ids}

		for time in times:
			id_pairs = []
			for i_id in node_ids:
				for j_id in node_ids:
					id_pairs.append((i_id, j_id))

			if self.verbose > 0:
				print("Gen data for time in _get_signals_borda")

			X = self._generate_data_for_rank(
				users_interaction_dict,
				id_pairs, 
				time)
					
			M = self.comparer.predict(X)
			M = M.reshape((len(node_ids), len(node_ids)))
			M[M == 0] = -1
			np.fill_diagonal(M, 0)

			counts = np.sum(M, axis=1)

			for uid, count in zip(node_ids, counts):
				borda_counts[uid].append(count)

		ret_dict = dict()

		for uid in ids:
			if uid in borda_counts.keys():
				ret_dict[uid] = (np.asarray(borda_counts[uid]) 
									+ len(node_ids)) / (2 * len(node_ids))
			else:
				ret_dict[uid] = np.zeros(len(times))

		return ret_dict

if __name__ == "__main__":
	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	tspr = TimeSeriesPairwiseRanker(None, window=21, verbose=147)

	# X, y = tspr._generate_data(interaction_dict, survey_dict)
	uid = 10060
	uid_interactions = interaction_dict[uid]
	survey_time = 1327890900

	node_ids = list(uid_interactions.keys())

	id_pairs = []
	for i_id in node_ids:
		for j_id in node_ids:
			id_pairs.append((i_id, j_id))

	Xr = tspr._generate_data_for_rank(uid_interactions, id_pairs, survey_time)

