import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd

import dask
from dask.distributed import Client, LocalCluster

import rbo  # https://github.com/changyaochen/rbo

from rankers import jaccard_similarity, kendal_tau


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


def get_signals(start_times, observation_times, mu, theta, forget_type, forget_intensity):

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


@dask.delayed
def evaluate_for_node(events, surveys, L_vals, mu_vals, theta_vals, forget_types):
	return_matrix = []

	survey_times = sorted(list(surveys.keys()))
	node_ids = np.asarray(list(events.keys()))
	node_events = [sorted(events_mat[:, 2]) for events_mat in events.values()]

	for L, mu, theta, forget_type in product(L_vals, mu_vals, theta_vals, forget_types):
		if theta >= mu:
			continue
		forget_intensity = get_forget_intensity(L, mu, theta, forget_type)

		signal_strengths = np.asarray(
			[get_signals(event_times, survey_times, mu, theta, forget_type, forget_intensity)
				for event_times in node_events]
		)

		for i in range(len(survey_times)):
			top_n = list(surveys[survey_times[i]].values())
			cogsnet_top_n = node_ids[(-signal_strengths[:, i]).argsort()[:len(top_n)]]

			return_matrix.append(
				[L, mu, theta, forget_type, 
					jaccard_similarity(top_n, cogsnet_top_n),
					rbo.RankingSimilarity(top_n, cogsnet_top_n).rbo(),
					kendal_tau(top_n, cogsnet_top_n)
				])

	return return_matrix


def evaluate_model_params(edge_dict, interaction_dict, survey_dict,
                          L_vals, mu_vals, theta_vals, forget_types):
	"""
	Given interaction data, survey data, and lists of parameters to check,
		creates a dataframe with a row for each combination of parameters.
		The combination of parameters will have the average jaccard similarity
		and rank-biased overlap (RBO) across all surveys.

	Creates a list of dask delayed processes, each of which handle one node who
		has survey data.  

	Return example:
		     L   mu  theta forget_func  jaccard_sim       rbo
		0  1.0  0.1   0.05         exp     0.240194  0.298805
		1  2.0  0.1   0.05         exp     0.286562  0.327788
	"""
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

	res_df = pd.DataFrame(
		res_matrix, columns=['L', 'mu', 'theta', 'forget_func', 'jaccard_sim', 'rbo', 'kendall_tau'])
	res_df[['L', 'mu', 'theta', 'jaccard_sim', 'rbo', 'kendall_tau']
        ] = res_df[['L', 'mu', 'theta', 'jaccard_sim', 'rbo', 'kendall_tau']].astype(float)
	res_df.L = res_df.L / 24

	return res_df


if __name__ == "__main__":
	# Create dask cluster
	cluster = LocalCluster(n_workers=100, dashboard_address=':8765')
	client = Client(cluster)

	print("loading data")

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
	# L_vals = np.asarray([1, 4, 7, 14, 21, 28, 60, 90, 120, 180, 365, 460, 730]) * 24
	# mu_vals = np.linspace(.002, 1, 60)
	# theta_vals = np.linspace(.001, 1, 60)
	# forget_types = ['exp']

	# run 8
	# L_vals = np.asarray(range(1, 29)) * 24
	# mu_vals = np.linspace(.00001, .1, 30)
	# theta_vals = np.linspace(.00001, .1, 30)
	# forget_types = ['exp']

	# final
	L_vals = np.asarray(range(1, 29)) * 24
	mu_vals = np.asarray(list(
		set([0.09310414, 0.08620828, 0.09655207, 0.1, 0.02414552,
            	0.08965621, 0.02759345, 0.03104138, 0.02069759, 0.03448931]
		).union( [0.04138517, 0.01380172, 0.03448931, 0.0448331, 0.03104138,
					0.02759345, 0.01724966, 0.02069759, 0.02414552, 0.03793724]
		).union([0.002, 0.08657627, 0.10349153, 0.05274576, 0.06966102,
					0.03583051, 0.01891525]
		).union([0.03583051, 0.06966102, 0.05274576, 0.12040678, 0.08657627,
					0.01891525, 0.10349153]
		).union([0.25, 0.3, 0.2, 0.15, 0.05, 0.1])))
	theta_vals = np.asarray(list(
		set([0.08620828, 0.07931241, 0.09310414, 0.09655207, 0.02069759,
               	0.08965621, 0.02414552, 0.02759345, 0.08276034, 0.07586448,
               	0.01724966, 0.03104138]
		).union([0.03793724, 0.01035379, 0.03104138, 0.04138517, 0.02759345,
       				0.02414552, 0.01380172, 0.01724966, 0.02069759, 0.03448931]
        ).union([0.001, 0.08566102, 0.03486441, 0.10259322, 0.0179322,
                            0.06872881, 0.05179661]
        ).union([0.03486441, 0.06872881, 0.0179322, 0.05179661, 0.11952542,
                            0.08566102, 0.001, 0.10259322]
        ).union([1.33333333e-01, 6.66666667e-02, 2.33333333e-01, 2.66666667e-01,
                            2.00000000e-01, 1.00000000e-01, 1.66666667e-01, 3.33333333e-02,
                            1.00000000e-04])))
	forget_types = ['exp']

	# Preform grid search to create dataframe of parameters combination and
	# their respective performances
	start_time = time.time()

	res_df = evaluate_model_params(edge_dict, interaction_dict, survey_dict,
                                 	L_vals, mu_vals, theta_vals, forget_types)

	print(time.time() - start_time)

	# Format and save results
	mean_df = res_df.groupby(['L', 'mu', 'theta', 'forget_func']).mean().reset_index()
	mean_df.to_csv(os.path.join('results', 'mean_df.csv'))
	med_df = res_df.groupby(['L', 'mu', 'theta', 'forget_func']).median().reset_index()
	med_df.to_csv(os.path.join('results', 'med_df.csv'))

	client.close()
	cluster.close()
