import os 
from datetime import datetime

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from run_cogsnet import get_forget_intensity, get_signals, jaccard_similarity

if __name__ == "__main__":
	mu = .2
	theta = .1
	L = 864000 / 3600
	forget_type = 'exp'

	interactions = pd.read_pickle("data/interaction_dict.pkl")
	surveys = pd.read_pickle("data/survey_dict.pkl")

	target_id = 10060

	survey_times = sorted(list(surveys[target_id].keys()))
	forget_intensity = get_forget_intensity(L, mu, theta, forget_type)

	edge_ids = np.asarray(list(interactions[target_id].keys()))
	signals = []

	for edge in edge_ids:
		event_times = sorted(interactions[target_id][edge][:, 2])

		edge_sigs = get_signals(event_times, survey_times, mu, theta, 
								forget_type, forget_intensity)

		signals.append(edge_sigs)

	signals = np.stack(signals)

	top20s = []

	for survey_num in range(len(survey_times)):
		top_20_inds = np.argsort(-signals[:, survey_num])[:20]

		res_dict = {'rank': list(range(len(top_20_inds))),
					'edge_id': edge_ids[top_20_inds],
					'signal': signals[top_20_inds, survey_num]}

		res_df = pd.DataFrame(res_dict)

		top20s.append(res_df)

		print(res_df)
