import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rankers import PairwiseRanker
from comparers import Comparer, OnlyDiffSklearnClassifierComparer
from comparers import SklearnClassifierComparer, DiffSklearnClassifierComparer


N_SPLITS = 5
RAND_SEED = 147
RECORD_RESULTS = True


with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
	interaction_dict = pickle.load(pkl)

with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
	survey_dict = pickle.load(pkl)

surveys = []

for respondant_id, survey_times in survey_dict.items():
	for time in survey_times:
		surveys.append((respondant_id, time))

surveys = np.asarray(surveys)

k_fold = KFold(N_SPLITS, shuffle=True, random_state=RAND_SEED)

ranker_res = []

for train_inds, test_inds in k_fold.split(surveys):

	# Split test and train surveys
	surveys_train = surveys[train_inds]
	surveys_test = surveys[test_inds]

	survey_dict_train = {resp: dict() for resp, _ in surveys_train}
	for resp, survey_time in surveys_train:
		survey_dict_train[resp][survey_time] = survey_dict[resp][survey_time]

	survey_dict_test = {resp: dict() for resp, _ in surveys_test}
	for resp, survey_time in surveys_test:
		survey_dict_test[resp][survey_time] = survey_dict[resp][survey_time]

	# Create and fit ranker
	ranker = PairwiseRanker(SklearnClassifierComparer(
			Pipeline([
				('scale', StandardScaler()),
				('classify', AdaBoostClassifier(n_estimators=100))],
				verbose=True
			),
         	desc="std_scaler+AdaBoost n=100"
        ),
		verbose=1)

	ranker.fit(interaction_dict, survey_dict_train)

	ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
	ranker_res[-1]['desc'] = str(ranker)

	ranker.rank_method = 'borda'
	ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
	ranker_res[-1]['desc'] = str(ranker)

	print(ranker_res)

# compile results
res_df = pd.DataFrame(ranker_res).groupby('desc').mean().reset_index()

print(res_df)

# update dataframe of all test results
if RECORD_RESULTS:
	all_res = pd.read_pickle("pairwise_res.pkl")
	all_res = all_res.append(res_df, ignore_index=True)
	print(all_res)
	all_res.to_pickle("pairwise_res.pkl")