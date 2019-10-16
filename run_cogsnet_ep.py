import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd

import dask
from dask.distributed import Client, LocalCluster

import rbo  # https://github.com/changyaochen/rbo

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


@dask.delayed
def get_signals(start_times, observation_times, mu, theta, forget_type, forget_intensity):
	# start_times = sorted(start_times) #remove sorts for speedup
	# observation_times = sorted(observation_times)

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

		current_signal = mu

	if obs_ind >= total_obs:
		return ret_values

	for i in range(1, len(start_times)):
		while start_times[i] > observation_times[obs_ind]:
			val_at_obs = current_signal * forget_func(
							forget_type,
							(observation_times[obs_ind] - start_times[i-1]) / 3600,
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
							(start_times[i] - start_times[i-1]) / 3600,
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
def evaluate_for_node(events, surveys, L_vals, mu_vals, theta_vals, forget_types):
	return_matrix = []

	for L, mu, theta, forget_type in product(L_vals, mu_vals, theta_vals, forget_types):
		if theta >= mu:
			continue
		forget_intensity = get_forget_intensity(L, mu, theta, forget_type)

		node_ids = np.asarray(list(events.keys()))
		survey_times = list(surveys.keys())

		signal_strengths = [
				get_signals(events_mat[:, 2], survey_times, mu, theta, forget_type, forget_intensity)
				for events_mat in events.values()
			]

		global s
		signal_strengths = np.asarray(dask.compute(signal_strengths)[0])
		s = signal_strengths

		for i in range(len(survey_times)):
			top_n = list(surveys[survey_times[i]].values())
			cogsnet_top_n = node_ids[(-signal_strengths[:, i]).argsort()[:len(top_n)]]

			return_matrix.append(
				[L, mu, theta, forget_type, 
					jaccard_similarity(top_n, cogsnet_top_n),
					rbo.RankingSimilarity(top_n, cogsnet_top_n).rbo()
				])

	return return_matrix


def evaluate_model_params(edge_dict, interaction_dict, survey_dict,
                          L_vals, mu_vals, theta_vals, forget_types):
	res_matrix = []

	n = 1
	for participant_id in survey_dict.keys():
		print(n)
		if (participant_id in edge_dict.keys()):
			res_matrix.append(evaluate_for_node(
				interaction_dict[participant_id],
				survey_dict[participant_id],
				L_vals,
				mu_vals,
				theta_vals,
				forget_types
			))
		n += 1

	res_matrix = np.vstack(dask.compute(res_matrix)[0])

	return res_matrix


if __name__ == "__main__":
	cluster = LocalCluster(n_workers=90, dashboard_address=':8766')
	client = Client(cluster)

	with open(os.path.join("data", "edge_dict.pkl"), 'rb') as pkl:
		edge_dict = pickle.load(pkl)

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)
	
	# create values which will be used in grid search

	# run 1
	# L_vals = np.asarray(range(1, 365, 2)) * 24
	# mu_vals = np.asarray([.4, .6, .8])
	# theta_vals = np.asarray([.05, .1, .3, .5])
	# forget_types = ['pow', 'exp']

	# # run 2
	# # L_vals = np.asarray(range(250, 450, 2)) * 24
	# L_vals = np.asarray([350, 400]) * 24
	# mu_vals = np.linspace(.5, .7, 20)
	# theta_vals = np.linspace(.1, .4, 20)
	# forget_types = ['exp']

	# # run 3
	# # L_vals = np.asarray(range(250, 450, 2)) * 24
	# L_vals = np.asarray(
	# 	[10, 50, 100, 147, 166, 181, 200, 261, 265, 255, 253, 251, 263, 279, 
	# 		277, 271, 269, 275, 273, 299,
	# 		297, 301, 291, 289, 293, 303, 307, 305, 295, 319, 309, 317, 311,
	# 		313, 315, 325, 323, 321, 327, 331, 329, 333, 359, 357, 345, 347,
	# 		343, 339, 355, 363, 337, 353, 341, 361, 349, 335, 351, 380, 390, 
	# 		400, 410, 420, 450, 460]
	# 	) * 24
	# mu_vals = np.asarray([.2, .25, .3, .35, .4, .45, .5, .6, .7])
	# theta_vals = np.linspace(.1, .4, 10)
	# forget_types = ['exp']

	# # run 3
	L_vals = np.asarray([460]) * 24
	mu_vals = np.asarray([.2])
	theta_vals =np.asarray([.166667])
	forget_types = ['exp']

	# # run 4
	# L_vals = np.asarray(
	# 	[359, 321, 333, 315, 335, 339, 329, 325, 319, 313, 331, 337, 317,
	# 		327, 323, 295, 293, 297, 291, 311, 100, 341, 345, 343, 351, 355,
	# 		353, 289, 361, 349, 347, 200, 420, 390, 166, 357, 363, 380, 147,
	# 		263, 265, 261, 269, 181, 400, 410, 277, 279, 275, 271, 273, 299,
	# 		255, 460, 450, 303, 307, 301, 305, 253, 309, 251,
	# 		470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]
	# 	) * 24
	# mu_vals = np.asarray([.15, .17, .19, .2, .21, .23, .24, .25])
	# theta_vals = np.linspace(0, 2.5, 10)[1:]
	# forget_types = ['exp']

	# # run 5
	# L_vals = np.asarray([7, 14, 21, 60, 90, 120, 180, 365, 460]) * 24
	# mu_vals = np.asarray([.2, .3, .4, .5, .6, .7, .8, .9, 1.0])
	# theta_vals =np.asarray([.1, .166667, .2, .3, .4, .5, .6])
	# forget_types = ['exp', 'pow']
	# run 5
	# L_vals = np.asarray([7, 460]) * 24
	# mu_vals = np.asarray([.2,  1.0])
	# theta_vals =np.asarray([.1,  .6])
	# forget_types = ['exp', 'pow']


	start_time = time.time()

	results = evaluate_model_params(edge_dict, interaction_dict, survey_dict,
                                 	L_vals, mu_vals, theta_vals, forget_types)

	print(time.time() - start_time)

	res_df = pd.DataFrame(results, columns=['L', 'mu', 'theta', 'forget_func', 'jaccard_sim', 'rbo'])
	res_df[['L', 'mu', 'theta', 'jaccard_sim', 'rbo']
        ] = res_df[['L', 'mu', 'theta', 'jaccard_sim', 'rbo']].astype(float)
	res_df.L = res_df.L / 24

	mean_df = res_df.groupby(['L', 'mu', 'theta', 'forget_func']).mean().reset_index()
	# mean_df.to_csv(os.path.join('results', 'mean_df.csv'))
	# med_df = res_df.groupby(['L', 'mu', 'theta', 'forget_func']).median().reset_index()
	# med_df.to_csv(os.path.join('results', 'med_df.csv'))

	client.close()
	cluster.close()
