import os
from datetime import datetime
from collections import Counter
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import rbo  # https://github.com/changyaochen/rbo

from rankers import *
from comparers import *
from rankers_util import load_keras_ranker


if __name__ == "__main__":
	# Load saved ts rankers
	no_scaling_lstm = load_keras_ranker(os.path.join('trained_models_netsense',
													 'best_avg_lstm_no_scaling'))
	no_scaling_lstm.rank_method = 'tau'
	no_scaling_lstm.desc_str = 'LSTM no scaling'

	no_scaling_lstm_b = load_keras_ranker(os.path.join('trained_models_netsense',
													 'best_avg_lstm_no_scaling'))
	no_scaling_lstm_b.rank_method = 'borda'
	no_scaling_lstm_b.desc_str = 'LSTM no scaling (borda)'

	scaling_lstm = load_keras_ranker(os.path.join('trained_models_netsense',
													 'best_jaccard_lstm'))
	scaling_lstm.rank_method = 'borda'
	scaling_lstm.desc_str = 'LSTM with scaling'                                         

	# Rankers to evaluate
	rankers = [
		VolumeRanker(),
		# WindowedVolumeRanker(window_size=21),
		# WindowedVolumeRanker(window_size=150),
		HawkesRanker(beta=1.697e-7),
		# HawkesRanker(beta=2.2209e-7), # HawkesRankerL(L=120, p=.1)
		CogSNetRanker(L=12, mu=.0189153, theta=.0179322, 
						desc_str="CogSNet (best overall)"),
		# CogSNet2Ranker(L=250, mu=.01725),
		# CogSNet2Ranker(L=250, mu=.1725),
		scaling_lstm,
		no_scaling_lstm,
		no_scaling_lstm_b
	]

	# edges to plot
	plot_all = False		# if true plot_n ignored
	plot_all_true = True	# if true will plot (all ground true) union (top plot_n)
	plot_n = 40

	# load data
	interactions = pd.read_pickle("data/interaction_dict.pkl")
	surveys = pd.read_pickle("data/survey_dict.pkl")

	ids_and_n_edges = [(k, len(interactions[k])) for k in surveys.keys()]
	# id_to_plot = 15548
	# id_to_plot = 75466
	# id_to_plot = 80058
	# id_to_plot = 86727
	# id_to_plot = 62555
	# id_to_plot = 30076
	id_to_plot = 69669
	
	fig = plot_rankers_grid(rankers, interactions, surveys, id_to_plot, 
							verbose=True, n_samples=1000, 
							plot_all=plot_all, plot_all_true=plot_all_true,
							plot_top_n=plot_n)

	fig.savefig("vis_ts/{}_signals_{}_rankers".format(id_to_plot, len(rankers)),
			 	dpi=100)

	plt.close()
