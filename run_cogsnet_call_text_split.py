import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd

import dask
from dask.distributed import Client, LocalCluster

import rbo  # https://github.com/changyaochen/rbo


class ParamGrid(object):
	"""
	Will hold all values for each parameter and thus will be able to
		return all combinations of them to be evaluated 
	"""

	def __init__(self, L, mu_call, mu_text, theta, forget_func):
		self.L = L
		self.mu_call = mu_call
		self.mu_text = mu_text
		self.theta = theta
		self.forget_func = forget_func

	def get_grid(self):
		return product(self.L, self.mu_call, self.mu_text,
                 self.theta, forget_types)


def jaccard_similarity(list1, list2):
	s1 = set(list1)
	s2 = set(list2)
	return len(s1.intersection(s2)) / len(s1.union(s2))


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


def get_signals_2_class(start_times, event_types, observation_times,
                          mu_0, mu_1, theta, forget_type, forget_intensity):

	ret_values = []

	current_signal = 0
	obs_ind = 0
	total_obs = len(observation_times)

	if len(start_times) > 0:
		while start_times[0] > observation_times[obs_ind]:
			ret_values.append(current_signal)
			obs_ind += 1
			if obs_ind >= total_obs:
				return ret_values

		if event_types[0] == 0:
			current_signal = mu_0
		else:
			current_signal = mu_1

	if obs_ind >= total_obs:
		return ret_values

	for i in range(1, len(start_times)):
		while start_times[i] > observation_times[obs_ind]:
			val_at_obs = current_signal * forget_func(
                            forget_type,
                            (observation_times[obs_ind] -
                             start_times[i-1]) / 3600,
                            forget_intensity)

			if val_at_obs < theta:
				ret_values.append(0)
			else:
				ret_values.append(val_at_obs)

			obs_ind += 1

			if obs_ind >= total_obs:
				break

		if obs_ind >= total_obs:
				break

		decayed_signal = current_signal * forget_func(forget_type,
                                                (start_times[i] -
                                                 start_times[i-1]) / 3600,
                                                forget_intensity)
		if decayed_signal < theta:
			decayed_signal = 0

		if event_types[0] == 0:
			mu = mu_0
		else:
			mu = mu_1
		current_signal = mu + decayed_signal * (1 - mu)

	while obs_ind < total_obs:
		val_at_obs = current_signal * forget_func(
                    forget_type,
                    (observation_times[obs_ind] - start_times[-1]) / 3600,
                    forget_intensity)

		if val_at_obs < theta:
				ret_values.append(0)
		else:
			ret_values.append(val_at_obs)

		obs_ind += 1

	return ret_values


@dask.delayed
def evaluate_for_node(events, surveys, param_grid):
	return_matrix = []

	survey_times = sorted(list(surveys.keys()))
	node_ids = np.asarray(list(events.keys()))

	event_times = []
	event_types = []

	for events_mat in events.values():
		event_order = np.argsort(events_mat[:, 2])
		event_times.append(events_mat[event_order][:, 2])
		event_types.append(events_mat[event_order][:, 0])

	for L, mu_t, mu_c, theta, forget_type in param_grid.get_grid():
		if theta >= mu_t or theta >= mu_c:
			continue
		forget_intensity = get_forget_intensity(L, (mu_c + mu_t) / 2, theta, 
												forget_type)

		signal_strengths = np.asarray(
			[get_signals_2_class(times, types, survey_times,
					mu_t, mu_c, theta, forget_type, forget_intensity)
				for times, types in zip(event_times, event_types)]
		)

		for i in range(len(survey_times)):
			top_n = list(surveys[survey_times[i]].values())
			cogsnet_top_n = node_ids[(-signal_strengths[:, i]).argsort()[:len(top_n)]]

			return_matrix.append(
                            [L, mu_t, mu_c, theta, forget_type,
                             jaccard_similarity(top_n, cogsnet_top_n),
                                rbo.RankingSimilarity(
                                 top_n, cogsnet_top_n).rbo()
                             ])

	return return_matrix


def evaluate_model_params(edge_dict, interaction_dict, survey_dict, param_grid):
	"""
	Given interaction data, survey data, and lists of parameters to check,
		creates a dataframe with a row for each combination of parameters.
		The combination of parameters will have the average jaccard similarity
		and rank-biased overlap (RBO) across all surveys.

	Creates a list of dask delayed processes, each of which handle one node who
		has survey data.  

	Return example:
		     L   mu_t   mu_c  theta forget_func  jaccard_sim       rbo
		0  1.0    0.1    0.2   0.05         exp     0.240194  0.298805
		1  2.0    0.1    0.2   0.05         exp     0.286562  0.327788
	"""
	res_matrix = []

	n = 1
	for participant_id in survey_dict.keys():
		print(n)
		if (participant_id in edge_dict.keys()):
			res_matrix.append(evaluate_for_node(
				interaction_dict[participant_id],
				survey_dict[participant_id],
				param_grid
			))
		n += 1

	res_matrix = np.vstack(dask.compute(res_matrix)[0])

	res_df = pd.DataFrame(
				res_matrix, 
				columns=['L', 'mu_t', 'mu_c', 'theta', 'forget_func', 
							'jaccard_sim', 'rbo'])

	res_df[['L', 'mu_t', 'mu_c', 'theta', 'jaccard_sim', 'rbo']
        ] = res_df[['L', 'mu_t', 'mu_c', 'theta', 'jaccard_sim', 'rbo']].astype(float)
	res_df.L = res_df.L / 24

	return res_df


if __name__ == "__main__":
	# Create dask cluster
	cluster = LocalCluster(n_workers=70, dashboard_address=':8766')
	client = Client(cluster)

	# Load required dicts
	with open(os.path.join("data", "edge_dict.pkl"), 'rb') as pkl:
		edge_dict = pickle.load(pkl)

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	# create values which will be used in grid search

	# # # run 3 validation
	# L_vals = np.asarray([460]) * 24
	# mu_vals = np.asarray([.2])
	# theta_vals =np.asarray([.166667])
	# forget_types = ['exp']

	# run 6 val
	L_vals = np.asarray([1, 4, 7, 14, 21, 28, 60, 90, 120, 180, 365, 460, 730]) * 24
	mu_vals = np.linspace(.002, .9, 20)
	theta_vals = np.linspace(.001, .5, 20)
	forget_types = ['exp']

	param_grid = ParamGrid(L_vals, mu_vals, mu_vals,
                        theta_vals, forget_types)

	# Preform grid search to create dataframe of parameters combination and
	# their respective performances
	start_time = time.time()

	res_df = evaluate_model_params(edge_dict, interaction_dict, survey_dict,
                                	param_grid)

	print(time.time() - start_time)

	# Format and save results
	mean_df = res_df.groupby(
		['L', 'mu_t', 'mu_c', 'theta', 'forget_func']).mean().reset_index()
	mean_df.to_csv(os.path.join('results', 'mean_df_split.csv'))
	med_df = res_df.groupby(['L', 'mu', 'theta', 'forget_func']).median().reset_index()
	med_df.to_csv(os.path.join('results', 'med_df_split.csv'))

	client.close()
	cluster.close()
