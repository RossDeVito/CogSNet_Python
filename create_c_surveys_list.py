import os
import pickle

import numpy as np 
import pandas as pd

if __name__ == "__main__":
	with open(os.path.join("data", "survey_textcall_dict.pkl"), 'rb') as pkl:
		survey_dict = pickle.load(pkl)

	out_file = open(os.path.join("data", 
						"python-derived-survey-nodes-times-list.txt"),
					'w')

	for survey_respondant_id, responses in survey_dict.items():
		for survey_num, survey_time in enumerate(responses.keys()):
			for rank, answer_id in enumerate(responses[survey_time].values(), start=1):
				print(" {}  {:6}  {} {:2}   {} -1".format(
							survey_respondant_id,
							answer_id,
							survey_num,
							rank,
							survey_time
						), 
						file=out_file)







	print("-1", file = out_file)

	out_file.close()

