import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd

import dask
from dask.distributed import Client, LocalCluster


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


def get_signals(start_times, observation_times, event_type, 
				mu_0, forget_intensity_0, mu_1, forget_intensity_1, 
				theta, forget_type):
	start_times = sorted(start_times)  # remove sorts for speedup
	observation_times = sorted(observation_times)

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

		if event_type[0] == 0:
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


# @dask.delayed
def evaluate_for_node(events, surveys, param_grid):
	return_matrix = []

	for L, mu_t, mu_c, theta, forget_type in param_grid.get_grid():
		forget_intensity_c = get_forget_intensity(L, mu_c, theta, forget_type)
		forget_intensity_t = get_forget_intensity(L, mu_t, theta, forget_type)

		node_ids = np.asarray(list(events.keys()))
		survey_times = list(surveys.keys())

		signal_strengths = np.asarray(
			[get_signals(events_mat[:, 2], survey_times, events_mat[:, 1], 
							mu_c, forget_intensity_c,
                            mu_t, forget_intensity_t,
							theta, forget_type)
				for events_mat in events.values()]
		)

		for i in range(len(survey_times)):
			top_n = list(surveys[survey_times[i]].values())

			return_matrix.append([L, mu, theta, forget_type, jaccard_similarity(
				top_n,
				node_ids[(-signal_strengths[:, i]).argsort()[:len(top_n)]]
			)])

	return return_matrix


def evaluate_model_params(edge_dict, interaction_dict, survey_dict, param_grid):
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

	return res_matrix


if __name__ == "__main__":
	# cluster = LocalCluster(n_workers=24, dashboard_address=':8776')
	# client = Client(cluster)

	with open(os.path.join("data", "edge_dict.pkl"), 'rb') as pkl:
		edge_dict = pickle.load(pkl)

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	# create values which will be used in grid search

	# run 1
	# L_vals = np.asarray(range(1, 365, 2)) * 24
	L_vals = np.asarray([365, 400, 460])
	mu_vals = np.asarray([.4, .6, .8])
	theta_vals = np.asarray([.05, .1, .3, .5])
	forget_types = ['pow', 'exp']

	param_grid = ParamGrid(L_vals, mu_vals, mu_vals, 
							theta_vals, forget_types)

	start_time = time.time()

	results = evaluate_model_params(edge_dict, interaction_dict, survey_dict,
                                 	param_grid)

	print(time.time() - start_time)

	res_df = pd.DataFrame(
		results, columns=['L', 'mu', 'theta', 'forget_func', 'jaccard_sim'])
	res_df[['L', 'mu', 'theta', 'jaccard_sim']
        ] = res_df[['L', 'mu', 'theta', 'jaccard_sim']].astype(float)
	res_df.L = res_df.L / 24

	mean_df = res_df.groupby(
		['L', 'mu', 'theta', 'forget_func']).mean().reset_index()
	mean_df.to_csv(os.path.join('results', 'mean_df_cts.csv'))
	med_df = res_df.groupby(['L', 'mu', 'theta', 'forget_func']).median().reset_index()
	med_df.to_csv(os.path.join('results', 'med_df_cts.csv'))

	# client.close()
	# cluster.close()
