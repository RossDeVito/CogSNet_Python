import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd


def jaccard_similarity(list1, list2):
	s1 = set(list1)
	s2 = set(list2)
	return len(s1.intersection(s2)) / len(s1.union(s2))


def evaluate_for_node(events, surveys):
	similarities = []

	for survey_time in surveys.keys():
		canidate_event_counts = {k: len(v[np.asarray(v[:,2] <= survey_time)])
								for k, v in events.items() 
								if len(v[np.asarray(v[:, 2] <= survey_time)] > 0)}

		ordered_inds = (-np.asarray(list(canidate_event_counts.values()))).argsort()
		ordered_canidates = np.asarray(list(canidate_event_counts.keys()))[ordered_inds]

		survey_res = list(surveys[survey_time].values())
		selected_ids = ordered_canidates[:len(survey_res)]

		similarities.append(jaccard_similarity(survey_res, selected_ids))

	return similarities


def evaluate_freq_model(edge_dict, interaction_dict, survey_dict):
	similarities = []

	n = 1
	n_parts = len(survey_dict.keys())

	for participant_id in survey_dict.keys():
		print("{} / {}".format(n, n_parts))

		if (participant_id in edge_dict.keys()):
			similarities.extend(evaluate_for_node(
				interaction_dict[participant_id],
				survey_dict[participant_id]
			))
		n += 1

	return np.mean(similarities)


if __name__ == "__main__":
	with open(os.path.join("data", "edge_dict.pkl"), 'rb') as pkl:
		edge_dict = pickle.load(pkl)

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	start_time = time.time()

	result = evaluate_freq_model(edge_dict, interaction_dict, survey_dict)

	print(time.time() - start_time)

	print("Avg Jaccard similarity:\t{}".format(result))
