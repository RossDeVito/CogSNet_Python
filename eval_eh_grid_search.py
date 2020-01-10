import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import dask
from dask.distributed import Client, LocalCluster

from rankers import HawkesRanker, EnHawkesRanker


def get_beta(L, p):
	""" give L in days and p in percent of beta remaining after L days """
	return np.log(1/p) / (L * 3600 * 24)


@dask.delayed
def get_score_rows(L1, p, interaction_dict, survey_dict_test):
	res = []

	for L2 in range(L1, 366):
		print("\trunning L1={} L2={}".format(L1, L2), flush=True)
		ranker = EnHawkesRanker(L1, p, L2)

		score = ranker.score(interaction_dict, survey_dict_test)

		res.append([str(ranker), L1, p, L2, score['jaccard'],
				score['rbo'], score['kendall_tau']])

	print("completed L1={}".format(L1), flush=True)

	return res


if __name__ == "__main__":
	n_splits = 5
	rand_seed = 147

	print("creating cluster")

	# Create dask cluster
	cluster = LocalCluster(n_workers=45, dashboard_address=':8761')
	client = Client(cluster)

	print("loading data")

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	# with open(os.path.join("data", "reality_commons_interaction_dict.pkl"), 'rb') as pkl:
	# 	interaction_dict = pickle.load(pkl)

	# with open(os.path.join("data", "reality_commons_survey_dict.pkl"), 'rb') as pkl:
	# 	survey_dict = pickle.load(pkl)

	surveys = []

	for respondant_id, survey_times in survey_dict.items():
		for time in survey_times:
			surveys.append((respondant_id, time))

	surveys = np.asarray(surveys)

	k_fold = KFold(n_splits, shuffle=True, random_state=rand_seed)

	ranker_res = []

	for train_inds, test_inds in k_fold.split(surveys):
		# surveys_train = surveys[train_inds]
		# surveys_test = surveys[test_inds]

		surveys_train = surveys
		surveys_test = surveys

		survey_dict_train = {resp: dict() for resp, _ in surveys_train}
		for resp, survey_time in surveys_train:
			survey_dict_train[resp][survey_time] = survey_dict[resp][survey_time]

		survey_dict_test = {resp: dict() for resp, _ in surveys_test}
		for resp, survey_time in surveys_test:
			survey_dict_test[resp][survey_time] = survey_dict[resp][survey_time]

		for L1 in range(101, 366):
			print("L1: {}".format(L1))

			res = get_score_rows(L1, .5, interaction_dict, survey_dict_test)

			ranker_res.append(res)

		break

	res_matrix = np.vstack(dask.compute(ranker_res)[0])

	print("compute finished")

	# compile results
	res_df = pd.DataFrame(
		res_matrix, 
		columns=['string', 'L1', 'p', 'L2', 'jaccard', 'rbo', 'kendall_tau'])
	res_df.to_csv('en_hawkes_grid_search_101-365.csv')

	print(res_df)

	client.close()
	cluster.close()
