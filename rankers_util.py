from pathlib import Path
import os

from datetime import datetime
import copy
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import kendalltau
from keras.models import load_model
import joblib


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

					
def get_recency(events, survey_time, earliest_timestamp=1312617635):
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


def plot_rankers_grid(rankers, interactions, surveys, id_to_plot,
						plot_all=False, plot_all_true=True, plot_top_n=20, 
						n_samples=100, tfpn_scheme=2, 
						width_per=5, height_per=1.1, top_addl_height=2,
						tight_layout_rect_args=[0.01, 0.01, 1, .95],
						subplot_left=.05, subplot_right=.99, dpi=100,
						verbose=False):
	""" 
	Plots the signals of selected rankers though time and shows 
	true/false predictions
	"""

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

	# Get id display order
	id_and_count = sorted(Counter(interesting_ids).items(), 
							key=lambda item: item[1], 
							reverse=True)
	ordered_ids = [uid for uid,count in id_and_count]     

	if plot_all:
		edges_to_plot = ordered_ids
	elif plot_all_true:
		edges_to_plot = [e for e in ordered_ids if ((e in ground_true_edges)
												 or (e in ordered_ids[:plot_top_n]))]
	else:
		edges_to_plot = ordered_ids[:plot_top_n]

	fig, ax = plt.subplots(
		nrows=len(edges_to_plot),
		ncols=len(rankers),
		sharex=True,
		sharey=True,
		figsize=(len(rankers) * width_per, 
				 len(edges_to_plot) * height_per + top_addl_height)
	)
	fig.suptitle("{} Signals".format(id_to_plot), 
					size='xx-large', 
					weight='bold', 
					x=.25)

	# Plot for each ranker
	for ranker_ind, ranker in enumerate(rankers):
		if verbose:
			print("Plotting {}".format(ranker))

		signals = ranker.get_signals(
			edges_to_plot, 
			interactions[id_to_plot], 
			sample_times)

		ax_ind = 0

		for edge_id in edges_to_plot:

			for time, time_unix in zip(survey_times_as_datetimes, survey_times):
				if tfpn_scheme == 1:
					if edge_id in surveys[id_to_plot][time_unix].values():
						if edge_id in predictions[str(ranker)][time_unix]:
							ax[ax_ind][ranker_ind].plot(
								(time, time), (0, 1), 
								c='g', linewidth=3.5)
						else:
							ax[ax_ind][ranker_ind].plot(
								(time, time), (0, 1), 
								c='r', linewidth=3.5)
					else:
						if edge_id in predictions[str(ranker)][time_unix]:
							ax[ax_ind][ranker_ind].plot(
								(time, time), (0, 1), 
								'--', c='r', linewidth=3.5)
						else:
							ax[ax_ind][ranker_ind].plot(
								(time, time), (0, 1), 
								'--', c='g', linewidth=3.5)
				elif tfpn_scheme == 2:
					if edge_id in surveys[id_to_plot][time_unix].values():
						if edge_id in predictions[str(ranker)][time_unix]:
							ax[ax_ind][ranker_ind].plot(
								(time, time), (0, 1), 
								c='g', linewidth=3.5)
						else:
							ax[ax_ind][ranker_ind].plot(
								(time, time), (0, 1), 
								'--', c='r', linewidth=3.5)
					else:
						if edge_id in predictions[str(ranker)][time_unix]:
							ax[ax_ind][ranker_ind].plot(
								(time, time), (0, 1), 
								c='r', linewidth=3.5)
						else:
							ax[ax_ind][ranker_ind].plot(
								(time, time), (0, 1), 
								'--', c='g', linewidth=3.5)

			ax[ax_ind][ranker_ind].plot(sample_times_as_datetime, 
									signals[edge_id],
									c='b', 
									linewidth=2)

			ax[ax_ind][0].set_ylabel(edge_id, rotation=90)
			ax[0][ranker_ind].set_title(
				"{}\nJacc = {:.3f}, RBO = {:.3f}, Kτ = {:.3f}".format(
                    str(ranker), 
					scores[str(ranker)]['jaccard'], 
					scores[str(ranker)]['rbo'],
					scores[str(ranker)]['kendall_tau']),
            	rotation=0)

			ax_ind += 1

	if verbose:
		print("Generating legend")

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

	if verbose:
		print("Formatting")

	fig.autofmt_xdate()
	fig.set_tight_layout({"rect": tight_layout_rect_args})
	fig.subplots_adjust(left=subplot_left, right=subplot_right)

	if verbose:
		print("Complete")

	return fig

def save_keras_ranker(ranker, dir_name, path='', overwrite=True):
	"""
	Save trained elements of a keras based model so that they can be reused

	Args:
		ranker: keras based times series classification based ranker 
			to save
		path: path to dir which will contain dir with ranker elements
		dir_name: name of dir in location path that will contain 
			ranker elements
		overwrite: if true will replace ranker already stored in location
	"""

	save_dir = os.path.join(path, dir_name)
	Path(save_dir).mkdir(parents=True, exist_ok=overwrite)

	comparer = ranker.comparer
	ranker.comparer = None

	pd.to_pickle(ranker, os.path.join(save_dir, 'ranker.pkl'))

	model = comparer.model
	scaler = comparer.scaler
	comparer.model = None
	comparer.scaler = None
	comparer.callbacks = 'removed to save. Should be saved as part of model'

	pd.to_pickle(comparer, os.path.join(save_dir, 'comparer.pkl'))
	model.save(os.path.join(save_dir, 'model.h5'))
	joblib.dump(scaler, os.path.join(save_dir, 'scaler.gz'))

	comparer.model = model
	comparer.scaler = scaler

	ranker.comparer = comparer

def load_keras_ranker(path):
	model = load_model(os.path.join(path, 'model.h5'))
	scaler = joblib.load(os.path.join(path, 'scaler.gz'))
	comparer = pd.read_pickle(os.path.join(path, 'comparer.pkl'))

	comparer.model = model
	comparer.scaler = scaler

	ranker = pd.read_pickle(os.path.join(path, 'ranker.pkl'))

	ranker.comparer = comparer

	return ranker