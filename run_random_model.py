import os
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd

import rbo  # https://github.com/changyaochen/rbo

from rankers import jaccard_similarity, kendal_tau


def jaccard_similarity(list1, list2):
	s1 = set(list1)
	s2 = set(list2)
	return len(s1.intersection(s2)) / len(s1.union(s2))


def evaluate_for_node(events, surveys):
	similarities = []

	for survey_time in surveys.keys():
		canidate_ids = [k for k, v in events.items() if np.any(v[:, 2] < survey_time)]
		np.random.shuffle(canidate_ids)

		survey_res = list(surveys[survey_time].values())
		selected_ids = canidate_ids[:len(survey_res)]

		similarities.append([
			jaccard_similarity(survey_res, selected_ids),
			rbo.RankingSimilarity(survey_res, selected_ids).rbo(),
			kendal_tau(survey_res, selected_ids)
		])

	return similarities


def evaluate_random_model(edge_dict, interaction_dict, survey_dict):
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
		
	return np.nanmean(np.array(np.stack(similarities), dtype=np.float), axis=0)


if __name__ == "__main__":
	# Load required dicts
	with open(os.path.join("data", "edge_dict.pkl"), 'rb') as pkl:
		edge_dict = pickle.load(pkl)

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	start_time = time.time()

	result = np.mean(
		[evaluate_random_model(edge_dict, interaction_dict, survey_dict)
			for i in range(1000)], axis=0)

	print(time.time() - start_time)

	print("Avg Jaccard similarity:\t{}".format(result[0]))
	print("Avg RBO:\t{}".format(result[1]))
	print("Avg Kendall Tau:\t{}".format(result[2]))
