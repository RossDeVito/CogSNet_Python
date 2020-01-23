import os
from datetime import datetime
from collections import Counter
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import rbo  # https://github.com/changyaochen/rbo

from rankers import *



if __name__ == "__main__":
	# Rankers to evaluate
	rankers = [
		VolumeRanker(),
		WindowedVolumeRanker(window_size=21),
		WindowedVolumeRanker(window_size=150)
	]

	# edges to plot
	plot_all = False		# if true plot_n ignored
	plot_all_true = True	# if true will plot (all ground true) union (top plot_n)
	plot_n = 10

	# True/false negative/positive coloring/line scheme (1 or 2)
	tfpn_scheme = 2

	# n time samples
	n_samples = 1000

	# load data
	interactions = pd.read_pickle("data/interaction_dict.pkl")
	surveys = pd.read_pickle("data/survey_dict.pkl")

	ids_and_n_edges = [(k, len(interactions[k])) for k in surveys.keys()]
	# id_to_plot = 15548
	# id_to_plot = 75466
	# id_to_plot = 69669 # no overlap, dont use
	id_to_plot = 80058
	# id_to_plot = 86727
	# id_to_plot = 62555
	# id_to_plot = 30076

	# Get times to sample
	all_event_times = np.hstack(
		[e[:, 2] for e in interactions[id_to_plot].values()])

	start_time = min(all_event_times) - 3600 * 24
	end_time = max(all_event_times) + 3600 * 24

	sample_times = np.linspace(start_time, end_time, n_samples).astype(int)
	sample_times_as_datetime = [
		datetime.utcfromtimestamp(t) for t in sample_times]

	#get survey times
	survey_times = list(surveys[id_to_plot].keys())
	survey_times_as_datetimes = [
		datetime.utcfromtimestamp(t) for t in survey_times]

	# get all edges mentioned in surveys
	edges_to_plot = set()
	for time in survey_times:
		edges_to_plot = edges_to_plot.union(
			surveys[id_to_plot][time].values())

	ground_true_edges = copy.deepcopy(edges_to_plot)	

	# get all edges that are predicted in top n by rankers and model scores
	predictions = dict()
	scores = dict()

	# List will collect ids that are true or predicted so that final plot
	#	will display most true/predicted edges
	interesting_ids = []

	# add true ids to interesting ids
	for s_time, rank_dict in surveys[id_to_plot].items():
		interesting_ids.extend([uid for uid in rank_dict.values()])

	for ranker in rankers:
		res = ranker.predict_and_score(interactions,
										{id_to_plot: surveys[id_to_plot]})

		predictions[str(ranker)] = res['predict'][id_to_plot]
		scores[str(ranker)] = res['score']

		interesting_ids.extend(
			list(np.hstack(list(res['predict'][id_to_plot].values())))
		)

	""" Plot for each ranker """

	# Get id display order
	id_and_count = sorted(Counter(interesting_ids).items(), 
							key=lambda item: item[1], 
							reverse=True)
	ordered_ids = [uid for uid,count in id_and_count]     

	if plot_all:
		edges_to_plot = ordered_ids
	elif plot_all_true:
		edges_to_plot = [e for e in ordered_ids if ((e in ground_true_edges)
												 or (e in ordered_ids[:plot_n]))]
	else:
		edges_to_plot = ordered_ids[:plot_n]

	fig, ax = plt.subplots(
		nrows=len(edges_to_plot),
		ncols=len(rankers),
		sharex=True,
		sharey=True,
		figsize=(len(edges_to_plot) * 4, len(edges_to_plot) * 1.1 + 2)
	)
	fig.suptitle("{} Signals".format(id_to_plot), 
					size='xx-large', 
					weight='bold', 
					x=.25)

	# Plot for each ranker
	for ranker_ind, ranker in enumerate(rankers):
		print("Plotting {}".format(ranker))

		signals = ranker.get_signals(
			edges_to_plot, 
			interactions[id_to_plot], 
			sample_times,
			norm_all=False)

		ax_ind = 0

		for edge_id in edges_to_plot:
			events = interactions[id_to_plot][edge_id]

			for time, time_unix in zip(survey_times_as_datetimes, survey_times):
				if tfpn_scheme == 1:
					if edge_id in surveys[id_to_plot][time_unix].values():
						if edge_id in predictions[str(ranker)][time_unix]:
							ax[ax_ind][ranker_ind].plot((time, time), (0, 1), c='g', linewidth=3.5)
						else:
							ax[ax_ind][ranker_ind].plot((time, time), (0, 1), c='r', linewidth=3.5)
					else:
						if edge_id in predictions[str(ranker)][time_unix]:
							ax[ax_ind][ranker_ind].plot((time, time), (0, 1), '--', c='r',
														linewidth=3.5)
						else:
							ax[ax_ind][ranker_ind].plot((time, time), (0, 1), '--', c='g',
														linewidth=3.5)
				elif tfpn_scheme == 2:
					if edge_id in surveys[id_to_plot][time_unix].values():
						if edge_id in predictions[str(ranker)][time_unix]:
							ax[ax_ind][ranker_ind].plot((time, time), (0, 1), c='g', linewidth=3.5)
						else:
							ax[ax_ind][ranker_ind].plot((time, time), (0, 1), '--', c='r', linewidth=3.5)
					else:
						if edge_id in predictions[str(ranker)][time_unix]:
							ax[ax_ind][ranker_ind].plot((time, time), (0, 1), c='r',
														linewidth=3.5)
						else:
							ax[ax_ind][ranker_ind].plot((time, time), (0, 1), '--', c='g',
														linewidth=3.5)

			ax[ax_ind][ranker_ind].plot(sample_times_as_datetime, 
									signals[edge_id],
									c='b', 
									linewidth=2)

			ax[ax_ind][0].set_ylabel(edge_id, rotation=90)
			ax[0][ranker_ind].set_title(
				"{}\nJacc = {:.4f}, RBO = {:.4f}".format(
                    str(ranker), 
					scores[str(ranker)]['jaccard'], 
					scores[str(ranker)]['rbo']),
            	rotation=0)

			ax_ind += 1

	# create legend
	if tfpn_scheme == 1:
		tp = Line2D([0], [0], color='g', linestyle='-',
				label='True Positive (on survey, no cogsnet)')
		fn = Line2D([0], [0], color='r', linestyle='-',
				label='False Negative (on survey, not no cogsnet)')
		tn = Line2D([0], [0], color='g', linestyle='--',
				label='True Negative (not on survey, not on cogsnet)')
		fp = Line2D([0], [0], color='r', linestyle='--',
				label='False Positive (not on survey, on cogsnet)')
	elif tfpn_scheme == 2:
		tp = Line2D([0], [0], color='g', linestyle='-',
				label='True Positive (on survey, no cogsnet)')
		fn = Line2D([0], [0], color='r', linestyle='--',
				label='False Negative (on survey, not no cogsnet)')
		tn = Line2D([0], [0], color='g', linestyle='--',
				label='True Negative (not on survey, not on cogsnet)')
		fp = Line2D([0], [0], color='r', linestyle='-',
				label='False Positive (not on survey, on cogsnet)')

	fig.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
            fancybox=False, ncol=2, handles=[tp, fp, fn, tn],
            fontsize='x-large')

	fig.autofmt_xdate()
	fig.set_tight_layout({"rect": [0, 0.01, 1, .95]})
	fig.savefig("vis_f/{}_signals_{}_rankers".format(id_to_plot, len(rankers)),
             	dpi=100)

	plt.close()
