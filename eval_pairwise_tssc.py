import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rankers import TimeSeriesPairwiseRanker
from comparers import TimeSeriesSummaryComparer


N_SPLITS = 5
RAND_SEED = 147
RECORD_RESULTS = True

PREPROCESSED_DATA_VERSION = "b7_w8_oT_sF_mC"
PREPROCESSED_DATA_DIR = os.path.join("summarized_time_series_data", 
										PREPROCESSED_DATA_VERSION)

ranker_res = []

for split in range(1, N_SPLITS + 1):

	X_train, y_train = pd.read_pickle(os.path.join(PREPROCESSED_DATA_DIR, 
										  "train_fold_{}.gzip".format(split)))

	X_test, y_test = pd.read_pickle(os.path.join(PREPROCESSED_DATA_DIR, 
										  "test_fold_{}.gzip".format(split)))

	# Create and fit ranker
	ranker = TimeSeriesPairwiseRanker(TimeSeriesSummaryComparer(
			RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1),
			desc="RandomForestClassifier(n_est=100)",
			verbose=1
		),
		verbose=1
	)

	ranker.fit(X=X_train, y=y_train)

	ranker_res.append(ranker.score(X=X_test, y=y_test))
	ranker_res[-1]['desc'] = str(ranker)

	ranker.rank_method = 'borda'
	ranker_res.append(ranker.score(X=X_test, y=y_test))
	ranker_res[-1]['desc'] = str(ranker)

	print(ranker_res)

# compile results
res_df = pd.DataFrame(ranker_res).groupby('desc').mean().reset_index()

print(res_df)

# update dataframe of all test results
if RECORD_RESULTS:
	all_res = pd.read_pickle("pairwise_tss_res.pkl")
	all_res = all_res.append(res_df, ignore_index=True)
	print(all_res)
	all_res.to_pickle("pairwise_res.pkl")
