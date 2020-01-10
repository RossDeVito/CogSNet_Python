import os
import pickle

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold

from rankers import VolumeRanker, WindowedVolumeRanker

if __name__ == "__main__":
	n_splits = 5
	rand_seed = 147

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

		# check range of window values
		for w in range(1, 366):
			ranker = WindowedVolumeRanker(w)
			ranker_res.append(ranker.score(interaction_dict, survey_dict_test))
			ranker_res[-1]['w'] = w

	# compile results
	res_df = pd.DataFrame(ranker_res).groupby('w').mean().reset_index()

	# res_df.to_csv('window_res.csv')
	print(res_df)

	melted_df = pd.melt(res_df, 
							id_vars=['w'], 
							value_vars=['jaccard', 'rbo', 'kendall_tau'],
							var_name='metric')

	sns.lineplot(x='w', y='value', hue='metric', data=melted_df, ci=None)
	plt.show()
	plt.savefig('windowed_res')
