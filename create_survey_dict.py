import os
import time
import datetime
from collections import defaultdict
import pickle

import pandas as pd

if __name__ == "__main__":
	survey_data = pd.read_csv(os.path.join("data", "survey-in.txt"),
								sep=';',
								names=['node', 'partner', 'semester_num',
										'datetime', 'survey_rank'])

	survey_data = survey_data.dropna()
	survey_data.semester_num = survey_data.semester_num.astype(int)
	survey_data.survey_rank = survey_data.survey_rank.astype(int)
	survey_data.partner = survey_data.partner.astype(int)

	survey_data['unix_time'] = survey_data.datetime.apply(lambda t: 
		time.mktime(datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timetuple())).astype(int)

	survey_dict = defaultdict(lambda: defaultdict(dict))

	for pair in survey_data.itertuples(index=False):
		survey_dict[pair.node][pair.unix_time][pair.survey_rank] = pair.partner

	# make all defaultdicts dicts
	for key in survey_dict.keys():
		survey_dict[key] = dict(survey_dict[key])
	survey_dict = dict(survey_dict)

	with open(os.path.join("data", "survey_dict.pkl"), 'wb') as pkl:
		pickle.dump(survey_dict, pkl, protocol=pickle.HIGHEST_PROTOCOL)
