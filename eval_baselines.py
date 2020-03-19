import os
import pickle

import numpy as np
import pandas as pd 

from sklearn.model_selection import KFold

from rankers import Ranker, RandomRanker, RecencyRanker, VolumeRanker

if __name__ == "__main__":
	n_splits = 5
	rand_seed = 147

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	with open(os.path.join("data", "new_weighted_survey_textcall_dict.pkl"), 'rb') as pkl:
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
		surveys_train = surveys[train_inds]
		surveys_test = surveys[test_inds]

		survey_dict_train = {resp: dict() for resp, _ in surveys_train}
		for resp, survey_time in surveys_train:
			survey_dict_train[resp][survey_time] = survey_dict[resp][survey_time]

		survey_dict_test = {resp: dict() for resp, _ in surveys_test}
		for resp, survey_time in surveys_test:
			survey_dict_test[resp][survey_time] = survey_dict[resp][survey_time]

		# First contact ranker
		ranker = Ranker()
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		# Random ranker
		ranker = RandomRanker()
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		# Recency ranker
		ranker = RecencyRanker()
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

		# Volume ranker
		ranker = VolumeRanker()
		ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
		ranker_res[-1]['desc'] = str(ranker)

	# compile results
	res_df = pd.DataFrame(ranker_res).groupby('desc').mean()

	print(res_df)
