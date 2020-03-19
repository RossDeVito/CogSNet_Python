import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from run_cogsnet import forget_func, get_signals, get_forget_intensity


if __name__ == "__main__":
	mu = .1
	theta = .05
	L = 20. * 24  # hours
	forget_type = 'exp'

	interactions = pd.read_pickle("data/interaction_dict.pkl")
	surveys = pd.read_pickle("data/survey_dict.pkl")

	target_id = 10060
	partner_id = 85596

	survey_times = list(surveys[target_id].keys())

	forget_intensity = get_forget_intensity(L, mu, theta, forget_type)
	
	r = get_signals(interactions[target_id][partner_id][:, 2], survey_times, mu,
	                theta, forget_type, forget_intensity)
