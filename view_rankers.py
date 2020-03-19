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


if __name__ == "__main__":
	# Rankers to evaluate
	rankers = [
		VolumeRanker(),
		# WindowedVolumeRanker(window_size=21),
		# WindowedVolumeRanker(window_size=150),
		# HawkesRanker(beta=1.697e-7),
		# HawkesRanker(beta=2.2209e-7), # HawkesRankerL(L=120, p=.1)
		CogSNetRanker(L=12, mu=.0189153, theta=.0179322, 
						desc_str="CogSNet (best overall)"),
		CogSNet2Ranker(L=250, mu=.01725),
		CogSNet2Ranker(L=250, mu=.1725),
	]
	# rankers = [
	# 	VolumeRanker(),
	# 	WindowedVolumeRanker(window_size=21),
	# 	WindowedVolumeRanker(window_size=150),
	# 	HawkesRanker(beta=1.697e-7),
	# 	HawkesRanker(beta=2.2209e-7), # HawkesRankerL(L=120, p=.1)
	# 	CogSNetRanker(L=12, mu=.0189153, theta=.0179322, 
	# 					desc_str="CogSNet (best overall)"),
	# 	CogSNetRanker(L=11, mu=.0344893, theta=.0333333,
	# 					desc_str="CogSNet (best Jaccard)"),
	# ]
	pairwise_rankers = []
	# 	(PairwiseRanker(
	# 		OnlyDiffSklearnClassifierComparer(
	# 			Pipeline([
	# 						('scale', StandardScaler()),
	# 						('classify', MLPClassifier(
	# 							(12,10,10,10,8,8,8,8,5,3), 
	# 							batch_size=128,
	# 							early_stopping=True,
	# 							verbose=True))
	# 					],
	# 					verbose=True),
	# 			desc="std_scaler+MLPClassifier((12,10,10,10,8,8,8,8,5,3))"),
	# 		verbose=1),
	# 	 ['tau', 'borda'],
	# 	 'Deep (10 layer) NN, Best Jaccard & Avg,'),
	# 	(PairwiseRanker(
	# 		DiffSklearnClassifierComparer(RandomForestClassifier(
	# 			n_estimators=1000, n_jobs=-1, verbose=2),
	# 			desc="RandomForest(n_estimators=1000)"), 
	# 		verbose=1),
	# 	 ['tau', 'borda'],
	# 	 'Random Forest (n=1000), Best RBO,'),
	# 	(PairwiseRanker(
	# 		OnlyDiffSklearnClassifierComparer(
	# 			Pipeline([
	# 						('scale', StandardScaler()),
	# 						('classify', MLPClassifier(
	# 							(10,8,5), 
	# 							batch_size=128,
	# 							early_stopping=True,
	# 							verbose=True))
	# 					],
	# 					verbose=True),
	# 			desc="std_scaler+MLPClassifier((10,8,5))"), 
	# 		verbose=1),
	# 	 ['tau', 'borda'],
	# 	 '3 layer NN, Best Kendall Tau,'),
	# ]

	# # edges to plot
	# plot_all = False		# if true plot_n ignored
	# plot_all_true = False	# if true will plot (all ground true) union (top plot_n)
	# plot_n = 20

	# load data
	interactions = pd.read_pickle("data/interaction_dict.pkl")
	surveys = pd.read_pickle("data/survey_dict.pkl")

	ids_and_n_edges = [(k, len(interactions[k])) for k in surveys.keys()]
	# id_to_plot = 15548
	# id_to_plot = 75466
	# id_to_plot = 80058
	# id_to_plot = 86727
	# id_to_plot = 62555
	id_to_plot = 30076

	# Fit models that need to be fit
	for model, rank_methods, name in pairwise_rankers:
		model = model.fit(interactions,
			{uid: data for uid, data in surveys.items() if uid != id_to_plot})

		for rank_meth in rank_methods:
			model_copy = copy.deepcopy(model)
			model_copy.rank_method = rank_meth
			model_copy.desc_str = "{} {}".format(name, rank_meth)

			rankers.append(model_copy)
	
	fig = plot_rankers_grid(rankers, interactions, surveys, id_to_plot, 
							verbose=True, n_samples=1000, plot_top_n=40)

	fig.savefig("vis_f/{}_signals_{}_rankers_c2".format(id_to_plot, len(rankers)),
             	dpi=100)

	plt.close()
