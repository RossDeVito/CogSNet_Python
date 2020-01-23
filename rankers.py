import os
from collections import defaultdict
import pickle
import warnings

import numpy as np
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import rbo  # https://github.com/changyaochen/rbo

from comparers import Comparer, SklearnClassifierComparer
from comparers import DiffSklearnClassifierComparer


def jaccard_similarity(list1, list2):
	s1 = set(list1)
	s2 = set(list2)
	union = s1.union(s2)
	if len(union) == 0:
		return 1.0
	return len(s1.intersection(s2)) / len(union)


def kendal_tau(list1, list2):
	""" list1 is ground truth, list2 is predictions """
	if len(list1) > 1 and len(list2) > 1:
		list1_encoding = [i + 1 for i in range(len(list1))]

		list2_encoding = []
		for i in list2:
			if i in list1:
				list2_encoding.append(list1.index(i) + 1)
			else:
				list2_encoding.append(np.inf)

		list1_encoding = np.array(list1_encoding)
		list2_encoding = np.array(list2_encoding)

		concordant = []
		discordant = []
		
		for i in range(len(list2_encoding) - 1):
			base = list2_encoding[i:]
			con = len(base[base > list2_encoding[i]])
			if con < len(list2_encoding) - (i + 1):
				discordant.append(1)
			else:
				discordant.append(0)
			concordant.append(con)
		C = np.sum(concordant)
		D = np.sum(discordant)
		return (C - D) * 1.000 / (C + D)
	else:
		return None


def get_volume(events, survey_time):
    """ Returns count of events before survey time """
    return len(events[events <= survey_time])


def get_volume_n_days_before(events, survey_time, n_days):
    return len(events[(events > (survey_time - n_days * 86400))
                      & (events <= survey_time)])

					
def get_recency(events, survey_time, earliest_timestamp=1189000931):
    """ Returns time delta between survey time and most recent event.

	For original dataset earlist_timestamp should be 1312617635
	For reality commons, earliest timestamp should be 1189000931

    If there are no events before or at survey time, returns time between
    earliest timestamp and survey time. earliest_timestamp should be the 
    earliest timestamp in the whole data set, not just the events in events.
    """
    events_at_time = events[events <= survey_time]

    if len(events_at_time) == 0:
        return survey_time - earliest_timestamp

    return survey_time - max(events_at_time)


def get_hawkes_signal(event_times, observation_time, beta):
	times_before_obs = event_times[event_times <= observation_time]
	time_deltas = observation_time - times_before_obs
	return np.sum(beta * np.exp(-beta * time_deltas))


def get_en_hawkes_signal(event_times, observation_time, beta, L2):
	times_before_obs = event_times[event_times <= observation_time]
	times_in_L1_window = times_before_obs[
		observation_time - (L2 * 3600 * 24) < times_before_obs]
	time_deltas = observation_time - times_in_L1_window
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

		ret_dict = dict()

		for uid in ids:
			ret_dict[uid] = np.asarray(counts[uid]) / max_events

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

		ret_dict = dict()

		for uid in ids:
			ret_dict[uid] = np.asarray(counts[uid]) / max_events

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

		ordered_inds = (
                    -np.asarray(list(hawkes_signals.values()))).argsort()
		ordered_canidates = np.asarray(
                    list(hawkes_signals.keys()))[ordered_inds]

		return ordered_canidates[:top_n]


class HawkesRankerL(Ranker):
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

	def __str__(self):
		return "PresetParamCogSNetRanker L={} mu={} theta={} f_type={}".format(
			self.L, self.mu, self.theta, self.forget_type
		)

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

	For the purpose of ranking, we will use τi as the score of any item i if tau
	ranking_method is used. Otherwise use borda count method.
	"""

	def __init__(self, comparer, rank_method='tau', verbose=0):
		""" comparer is a Comparer or subclass (see comparers.py) 
		
		Args:
			comparer: Comparer object
			rank_method: 'tau', 'borda', or 'wins'
			verbose: prints updates while fitting if > 0
		"""
		self.comparer = comparer
		self.verbose = verbose
		self.rank_method = rank_method

	def __str__(self):
		return "PairwiseRanker {}: {}".format(
			self.rank_method, str(self.comparer))

	def fit(self, interaction_dict, survey_dict):
		""" fits underlying comparer using all possible pairwise relationships
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
		""" Create the n x 1 feature vector to represent an individual at the
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
		if self.rank_method == 'wins':
			return self._rank_wins(data, top_n, survey_time)

		raise SystemExit("PairwiseRanker must have rank_method be 'tau', 'borda', or wins")

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

	def _rank_wins(self, data, top_n, survey_time):
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
		np.fill_diagonal(M, 0)

		counts = np.sum(M, axis=1)

		ordered_inds = (-counts).argsort()
		ordered_canidates = np.asarray(node_ids)[ordered_inds]

		return ordered_canidates[:top_n]


if __name__ == "__main__":
	""" main is used to test Ranker classes """

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)
	
	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	surveys = []

	for respondant_id, survey_times in survey_dict.items():
		for time in survey_times:
			surveys.append((respondant_id, time))

	surveys_train, surveys_test = train_test_split(surveys, test_size=.2, 
													random_state=147)

	survey_dict_train = {resp: dict() for resp, _ in surveys_train}
	for resp, survey_time in surveys_train:
		survey_dict_train[resp][survey_time] = survey_dict[resp][survey_time]

	survey_dict_test = {resp: dict() for resp, _ in surveys_test}
	for resp, survey_time in surveys_test:
		survey_dict_test[resp][survey_time] = survey_dict[resp][survey_time]

	# # Ranker
	# ranker = Ranker()

	# ranker.fit(interaction_dict, survey_dict)
	# ranker.fit()

	# rp = ranker.predict(interaction_dict, survey_dict)
	# rs = ranker.score(interaction_dict, survey_dict)


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

	# rec_ranker.fit(interaction_dict, survey_dict_train)
	# rec_ranker.fit()

	# recp = rec_ranker.predict(interaction_dict, survey_dict_test)
	# recs = rec_ranker.score(interaction_dict, survey_dict_test)


	# # HawkesRanker
	# hawkes_ranker = HawkesRanker(1.727784e-07)
	# hawkes_ranker_2 = HawkesRanker(beta=.00001)

	# hawkes_ranker.fit(interaction_dict, survey_dict)
	# hawkes_ranker_2.fit()

	# hrp = hawkes_ranker.predict(interaction_dict, survey_dict)
	# hrs = hawkes_ranker.score(interaction_dict, survey_dict)

	# hrp2 = hawkes_ranker_2.predict(interaction_dict, survey_dict)
	# hrs2 = hawkes_ranker_2.score(interaction_dict, survey_dict)


	# PairwiseRanker
	pw_ranker = PairwiseRanker(
		DiffSklearnClassifierComparer(
			RandomForestClassifier(n_estimators=10, n_jobs=-1,verbose=3),
		),
		rank_method='wins',
		verbose=1
	)
	# # pw_ranker = PairwiseRanker(Comparer())

	print("Starting fit")
	pw_ranker.fit(interaction_dict, survey_dict_train)

	print("Starting scoring")
	pws = pw_ranker.score(interaction_dict, survey_dict_test)

	print("Random Forest Classifier: {}".format(pws))

	# # CogSNetRanker
	# cogsnet_ranker = PresetParamCogSNetRanker(L=21.0, mu=0.018915, theta=0.017932,
    #                                        forget_type='exp')

	# cogsnet_ranker.fit(interaction_dict, survey_dict_train)
	# cogsnet_ranker.fit()

	# cnp = cogsnet_ranker.predict(interaction_dict, survey_dict_test)
	# cns = cogsnet_ranker.score(interaction_dict, survey_dict_test)
	# print("{}: {}".format(cogsnet_ranker, cns))


	# # BordaPairwiseRanker
	# bpw_ranker = BordaPairwiseRanker(CountComparer())

	# bpw_ranker.fit(interaction_dict, survey_dict)
	# bpw_ranker.fit()

	# # cnp = bpw_ranker.predict(interaction_dict, survey_dict)
	# bps = bpw_ranker.score(interaction_dict, survey_dict)
