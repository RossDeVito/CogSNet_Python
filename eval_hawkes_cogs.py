import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from rankers import HawkesRanker, CogSNet2Ranker, CogSNetRanker


def get_beta(L, p):
	""" give L in days and p in percent of beta remaining after L days """
	return np.log(1/p) / (L * 3600 * 24)


if __name__ == "__main__":
	n_splits = 5
	rand_seed = 147

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "weighted_survey_textcall_dict.pkl"), 'rb') as pkl:
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

	f = 1

	for train_inds, test_inds in k_fold.split(surveys):
		surveys_train = surveys[train_inds]
		surveys_test = surveys[test_inds]

		# surveys_train = surveys
		# surveys_test = surveys

		survey_dict_train = {resp: dict() for resp, _ in surveys_train}
		for resp, survey_time in surveys_train:
			survey_dict_train[resp][survey_time] = survey_dict[resp][survey_time]

		survey_dict_test = {resp: dict() for resp, _ in surveys_test}
		for resp, survey_time in surveys_test:
			survey_dict_test[resp][survey_time] = survey_dict[resp][survey_time]

		print("Hawkes 1")
		ranker = HawkesRanker(2.221e-07)
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		print("Hawkes 2")
		ranker = HawkesRanker(2.268e-07)
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		print("Hawkes 3")
		ranker = HawkesRanker(1.2e-07)
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		print("Hawkes 4")
		ranker = HawkesRanker(1.697e-07)
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		print("Cogs 1")
		ranker = CogSNetRanker(L=12, mu=.0189153, theta=.0179322)
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		print("Cogs 2")
		ranker = CogSNetRanker(L=11, mu=.0344893, theta=.0333333)
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		print("Cogs 3")
		ranker = CogSNetRanker(L=18, mu=.0189153, theta=.0179322)
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		print("Cogs 4")
		ranker = CogSNetRanker(L=25, mu=.0206976, theta=.0172497)
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		print("Finished fold {}".format(f))
		f += 1

		# break

	# compile results
	res_df = pd.DataFrame(ranker_res).groupby('desc').mean()

	print(res_df)
