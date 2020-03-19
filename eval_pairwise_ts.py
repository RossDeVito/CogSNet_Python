import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from rankers import TimeSeriesPairwiseRanker
from comparers import TimeSeriesComparer

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from rankers_util import save_keras_ranker
from tsc_models import *

# SPLIT PARAMS
N_SPLITS = 5
RAND_SEED = 147
RECORD_RESULTS = False

SAVE_MODEL = True
model_name = 'best_jaccard_lstm'

callbacks = [
	EarlyStopping(patience=12, verbose=1, restore_best_weights=True),
	ReduceLROnPlateau(factor=.5, patience=7, verbose=1)
]

with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
	interaction_dict = pickle.load(pkl)

with open(os.path.join("data", "weighted_survey_textcall_dict.pkl"), 'rb') as pkl:
	survey_dict = pickle.load(pkl)

# for uid in list(survey_dict.keys())[10:]:
# 	survey_dict.pop(uid)

surveys = []

for respondant_id, survey_times in survey_dict.items():
	for time in survey_times:
		surveys.append((respondant_id, time))

surveys = np.asarray(surveys)

k_fold = KFold(N_SPLITS, shuffle=True, random_state=RAND_SEED)

ranker_res = []

for train_inds, test_inds in k_fold.split(surveys):

	model = LSTM1(4)

	# # Split test and train surveys
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

	# Create and fit ranker
	ranker = TimeSeriesPairwiseRanker(
		TimeSeriesComparer(
			model,
			desc="LSTM1 bs=1024",
			batch_size=1024,
			epochs=200,
			callbacks=callbacks,
			verbose=2,
			validation_split=.1,
			n_workers=20
		),
		bin_size=21,
		other_feat=False,
		text_call_split=True,
		metric='count', # count, val, or both
		verbose=1)

	ranker.fit(interaction_dict, survey_dict_train)

	ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
	ranker_res[-1]['desc'] = str(ranker)

	ranker.rank_method = 'borda'
	ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
	ranker_res[-1]['desc'] = str(ranker)

	print(ranker_res)

	break

# compile results
res_df = pd.DataFrame(ranker_res).groupby('desc').mean().reset_index()

print(res_df)

if SAVE_MODEL:
	save_keras_ranker(ranker, path='trained_models', dir_name=model_name)

# update dataframe of all test results
if RECORD_RESULTS:
	all_res = pd.read_pickle("pairwise_ts_res.pkl")
	all_res = all_res.append(res_df, ignore_index=True)
	print(all_res)
	all_res.to_pickle("pairwise_ts_res.pkl")
