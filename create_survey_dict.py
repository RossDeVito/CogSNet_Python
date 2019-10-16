"""
Creates and pickles 2 dictionaries.

	survey_dict.pkl: for each node_id to complete a survey, contains
		information on when they completed surveys and their rankings
		on each survey

	survey_textcall_dict.pkl: same as survey_dict, but only contains
		survey responce node_ids that are in the call/text records and
		survey respondants who had survey responces that are in the 
		call/text records
"""

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

	# create full dict
	survey_dict = defaultdict(lambda: defaultdict(dict))

	for pair in survey_data.itertuples(index=False):
		survey_dict[pair.node][pair.unix_time][pair.survey_rank] = pair.partner

	# make all defaultdicts dicts
	for key in survey_dict.keys():
		survey_dict[key] = dict(survey_dict[key])
	survey_dict = dict(survey_dict)

	# create dict with only those w/ texts or calls 
	survey_dict_tc = defaultdict(lambda: defaultdict(dict))

	with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
		interaction_dict = pickle.load(pkl)

	for node_id in survey_dict.keys():
		edges_out = set(interaction_dict[node_id].keys())

		for time, rank_dict in survey_dict[node_id].items():
			filtered_rank_dict = {k: v for k, v in rank_dict.items()
									if v in edges_out}
			if len(filtered_rank_dict) > 0:
				survey_dict_tc[node_id][time] = filtered_rank_dict
		
		if len(survey_dict_tc[node_id]) == 0:
			survey_dict_tc.pop(node_id)

	# make all defaultdicts dicts
	for key in survey_dict_tc.keys():
		survey_dict_tc[key] = dict(survey_dict_tc[key])
	survey_dict_tc = dict(survey_dict_tc)

	with open(os.path.join("data", "survey_dict.pkl"), 'wb') as pkl:
		pickle.dump(survey_dict, pkl, protocol=pickle.HIGHEST_PROTOCOL)

	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'wb') as pkl:
		pickle.dump(survey_dict_tc, pkl, protocol=pickle.HIGHEST_PROTOCOL)
