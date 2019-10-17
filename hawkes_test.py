import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tick.plot import plot_hawkes_kernels
from tick.hawkes import HawkesExpKern

import rbo  # https://github.com/changyaochen/rbo


if __name__ == "__main__":
	# Load required dicts
	with open(os.path.join("data", "edge_dict.pkl"), 'rb') as pkl:
		edge_dict = pickle.load(pkl)

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	# Create input on which to fit model
	events = [y[:, 2]/3600 for x in interaction_dict.values() for y in x.values()]

	# for prelim testing
	events = events[100000:100030]
	start_time = min(np.hstack(events))

	# uncommenting this out results in a better plot
	# events = [e - start_time + 1 for e in events]

	hawkes_learner = HawkesExpKern(.01, verbose=True)
	hawkes_learner = hawkes_learner.fit(events, start_time)
	hawkes_learner.plot_estimated_intensity(events)
	plt.savefig("F")

	e1 = [events[0]]
	e2 = [events[7]]

	hl_1_then_2 = HawkesExpKern(.01, verbose=True)
	hl_2 = HawkesExpKern(.01, verbose=True)

	hl_1_then_2 = hl_1_then_2.fit(e1, start_time)
	hl_1_then_2 = hl_1_then_2.fit(e2, start_time)

	hl_2 = hl_2.fit(e2, start_time)

