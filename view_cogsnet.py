import os 
from datetime import datetime

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import rbo  # https://github.com/changyaochen/rbo

from run_cogsnet import get_forget_intensity, get_signals, jaccard_similarity

if __name__ == "__main__":
	mu = .05
	theta = .0333
	L_vals = np.asarray([7, 21, 60, 180, 365, 460]) * 24 # hours
	forget_type = 'exp'

	n_samples = 1000

	interactions = pd.read_pickle("data/interaction_dict.pkl")
	surveys = pd.read_pickle("data/survey_dict.pkl")

	ids_and_n_edges = [(k, len(interactions[k])) for k in surveys.keys()]
	# id_to_plot = 15548
	id_to_plot = 75466
	# id_to_plot = 69669 # no overlap, dont use
	# id_to_plot = 80058
	# id_to_plot = 86727
	# id_to_plot = 62555
	# id_to_plot = 30076

	all_event_times = np.hstack( 
		[e[:, 2] for e in interactions[id_to_plot].values()] )

	start_time = min(all_event_times) - 3600 * 24
	end_time = max(all_event_times) + 3600 * 24

	sample_times = np.linspace(start_time, end_time, n_samples).astype(int)
	sample_times_as_datetime = [datetime.utcfromtimestamp(t) for t in sample_times]

	survey_times = list(surveys[id_to_plot].keys())
	survey_times_as_datetimes = [datetime.utcfromtimestamp(t) for t in survey_times]

	# get all edges mentioned in surveys
	edges_to_plot = set()
	for time in survey_times:
		edges_to_plot = edges_to_plot.union(
			surveys[id_to_plot][time].values())

	# get all edges that are predicted in top n by CogSNet
	jacc_sims = dict()
	rbos = dict()
	nodes_predicted = dict()
	for L in L_vals:
		nodes_predicted[L] = dict()
		node_ids = np.asarray(list(interactions[id_to_plot].keys()))

		forget_intensity = get_forget_intensity(L, mu, theta, forget_type)
		signal_strengths = np.asarray(
			[get_signals(events_mat[:, 2], survey_times, mu, theta, forget_type, 
							forget_intensity)
				for events_mat in interactions[id_to_plot].values()]
		)

		for i, time in enumerate(survey_times):
			survey_top_n = list(surveys[id_to_plot][time].values())

			cogsnet_top_n = node_ids[(-signal_strengths[:, i]
			                          ).argsort()[:len(survey_top_n)]]

			jacc_sims[L/24] = jaccard_similarity(survey_top_n, cogsnet_top_n)
			rbos[L/24] = rbo.RankingSimilarity(survey_top_n, cogsnet_top_n).rbo()
			edges_to_plot = edges_to_plot.union(cogsnet_top_n)
			nodes_predicted[L][time] = cogsnet_top_n

	print("Jaccard Sims")
	print(jacc_sims)
	print("Rank Biased Overlap")
	print(rbos)

	# plot for each L
	edges_to_plot = list(edges_to_plot.intersection(interactions[id_to_plot].keys()))

	fig, ax = plt.subplots(
		nrows=len(edges_to_plot),
		ncols=len(L_vals),
		sharex=True,
		sharey=True,
		figsize=(len(L_vals) * 4, len(edges_to_plot) * 1.1 + 2)
	)
	fig.suptitle("{} CogSNet Signals (mu={}, theta={})".format(id_to_plot, mu, theta), 
					size='xx-large', weight='bold', x=.25)

	for L_ind, L in enumerate(L_vals):
		forget_intensity = get_forget_intensity(L, mu, theta, forget_type)

		ax_ind = 0

		for edge_id in edges_to_plot:
			events = interactions[id_to_plot][edge_id]

			signal = get_signals(events[:, 2], sample_times, mu, theta, 
							forget_type, forget_intensity)

			for time, time_unix in zip(survey_times_as_datetimes, survey_times):
				if edge_id in surveys[id_to_plot][time_unix].values():
					if edge_id in nodes_predicted[L][time_unix]:
						ax[ax_ind][L_ind].plot((time, time), (0, 1), c='g', linewidth=3.5)
					else:
						ax[ax_ind][L_ind].plot((time, time), (0, 1), c='r', linewidth=3.5)
				else:
					if edge_id in nodes_predicted[L][time_unix]:
						ax[ax_ind][L_ind].plot((time, time), (0, 1), '--', c='r', 
												linewidth=3.5)
					else:
						ax[ax_ind][L_ind].plot((time, time), (0, 1), '--', c='g', 
												linewidth=3.5)
			
			ax[ax_ind][L_ind].plot(sample_times_as_datetime, signal, c='b', linewidth=2)

			ax[ax_ind][0].set_ylabel(edge_id, rotation=90)
			ax[0][L_ind].set_title("L = {}\nJacc = {:.4f}, RBO = {:.4f}".format(
										L/24, jacc_sims[L/24], rbos[L/24]),
									rotation=0)

			ax_ind += 1

	# create legend
	tp = Line2D([0], [0], color='g', linestyle='-', 
						label='True Positive (on survey, no cogsnet)')
	fn = Line2D([0], [0], color='r', linestyle='-',
	                    label='False Negative (on survey, not no cogsnet)')
	tn = Line2D([0], [0], color='g', linestyle='--',
	                    label='True Negative (not on survey, not on cogsnet)')
	fp = Line2D([0], [0], color='r', linestyle='--',
                     	label='False Positive (not on survey, on cogsnet)')

	fig.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
            fancybox=False, ncol=2, handles=[tp, fp, fn, tn],
		   fontsize='x-large')
	
	fig.autofmt_xdate()
	fig.set_tight_layout({"rect": [0, 0.01, 1, .95]})
	fig.savefig("vis/{}_signals_{}_{}".format(id_to_plot, int(mu * 100), int(theta * 100)), 
				dpi=500)
		



