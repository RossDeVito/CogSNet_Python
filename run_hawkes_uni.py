import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd

import dask
from dask.distributed import Client, LocalCluster

from tick.hawkes import HawkesExpKern

import rbo  # https://github.com/changyaochen/rbo

from run_cogsnet import jaccard_similarity


def get_signals(event_times, observation_times, beta):
	ret_values = []

	event_times = np.asarray(event_times)

	global s
	s = event_times
	global o
	o = observation_times
	global d 
	d = beta

	for obs_time in observation_times:
		times_before_obs = event_times[event_times < obs_time]
		time_deltas = obs_time - times_before_obs
		ret_values.append(
			np.sum(beta * np.exp( -beta * time_deltas))
		)

	return ret_values


@dask.delayed
def evaluate_for_node(events, surveys, decays):
	return_matrix = []

	survey_times = sorted(list(surveys.keys()))
	node_ids = np.asarray(list(events.keys()))
	node_events = [sorted(events_mat[:, 2]) for events_mat in events.values()]

	for decay in decays:
		signal_strengths = np.asarray(
			[get_signals(event_times, survey_times, decay)
				for event_times in node_events]
		)

		for i in range(len(survey_times)):
			top_n = list(surveys[survey_times[i]].values())
			cogsnet_top_n = node_ids[(-signal_strengths[:, i]).argsort()[:len(top_n)]]

			return_matrix.append(
				[decay, 
					jaccard_similarity(top_n, cogsnet_top_n),
					rbo.RankingSimilarity(top_n, cogsnet_top_n).rbo()
				])

	return return_matrix


def evaluate_model_params(edge_dict, interaction_dict, survey_dict, decays):
	res_matrix = []

	n = 1
	for participant_id in survey_dict.keys():
		print(n)
		if (participant_id in edge_dict.keys()):
			res_matrix.append(evaluate_for_node(
				interaction_dict[participant_id],
				survey_dict[participant_id],
				decays
			))
		n += 1

	res_matrix = np.vstack(dask.compute(res_matrix)[0])

	res_df = pd.DataFrame(
		res_matrix, columns=['beta', 'jaccard_sim', 'rbo'])
	res_df[['beta', 'jaccard_sim', 'rbo']
        ] = res_df[['beta', 'jaccard_sim', 'rbo']].astype(float)

	return res_df


if __name__ == "__main__":
	# Create dask cluster
	cluster = LocalCluster(n_workers=50, dashboard_address=':8766')
	client = Client(cluster)

	# Load required dicts
	with open(os.path.join("data", "edge_dict.pkl"), 'rb') as pkl:
		edge_dict = pickle.load(pkl)

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	# Decays to evaluate
	# decays = [.0001, .001, .01, .1]
	decays = np.random.beta(.01, 10, 200)

	start_time = time.time()

	res_df = evaluate_model_params(edge_dict, interaction_dict, survey_dict,
                                 	decays)

	print(time.time() - start_time)

	# Format and save results
	mean_df = res_df.groupby(['beta']).mean().reset_index()
	mean_df.to_csv(os.path.join('results', 'mean_df_uh.csv'))
	med_df = res_df.groupby(['beta']).median().reset_index()
	med_df.to_csv(os.path.join('results', 'med_df_uh.csv'))

	client.close()
	cluster.close()
