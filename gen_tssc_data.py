import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from rankers import TimeSeriesPairwiseRanker
from comparers import TimeSeriesSummaryComparer


if __name__ == "__main__":
	n_splits = 5
	rand_seed = 147

	save_dir = os.path.join("summarized_time_series_data",
							"b7_w8_oT_sF_mC")

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	surveys = []

	for respondant_id, survey_times in survey_dict.items():
		for time in survey_times:
			surveys.append((respondant_id, time))

	surveys = np.asarray(surveys)

	k_fold = KFold(n_splits, shuffle=True, random_state=rand_seed)

	split_n = 1

	for train_inds, test_inds in k_fold.split(surveys):
		surveys_train = surveys[train_inds]
		surveys_test = surveys[test_inds]

		survey_dict_train = {resp: dict() for resp, _ in surveys_train}
		for resp, survey_time in surveys_train:
			survey_dict_train[resp][survey_time] = survey_dict[resp][survey_time]

		survey_dict_test = {resp: dict() for resp, _ in surveys_test}
		for resp, survey_time in surveys_test:
			survey_dict_test[resp][survey_time] = survey_dict[resp][survey_time]

		ranker = TimeSeriesPairwiseRanker(
			TimeSeriesSummaryComparer(None, verbose=0),
			verbose=1,
			bin_size=7,
			window=8,
			other_feat=True,
			text_call_split=False,
			metric="count"
		)

		X_test, y_test = ranker._generate_data(interaction_dict, survey_dict_test)

		X_test_sum = ranker.comparer._get_all_feats_summary(X_test)

		pd.to_pickle(
			[X_test_sum, np.asarray(y_test)], 
			os.path.join(save_dir, "test_fold_{}.gzip".format(split_n)))

		X_train, y_train = ranker._generate_data(interaction_dict, survey_dict_train)

		X_train_sum = ranker.comparer._get_all_feats_summary(X_train)

		pd.to_pickle(
			[X_train_sum, np.asarray(y_train)], 
			os.path.join(save_dir, "train_fold_{}.gzip".format(split_n)))

		split_n += 1


	print("COMPLETE")
