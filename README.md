# CogSNet Python
Python implementations of CogSNet model and models used for comparison

## Before Running Models
Before running a model, the files edge_dict.pkl, interaction_dict.pkl, and survey_dict.pkl must exist in the data directory. create_interaction_dicts.py creates the first two pickled dictionaries based on the data from telcodata.txt, which should be located in the data directory. create_survey_dict.py creates survey_dict.pkl based on data from survey-in.txt, which should also be located in the data directory.

## Models

### CogSNet
run_cogsnet.py applies the CogSNet model for all combinations of given parameters and then saves a csv of the results for each combination of parameters to the results directory.

### Random Sampling
run_random_model.py predicts top contacts by randomly selecting other nodes the node being predicted for had any interaction with in the time period from the start of the study to the time of the survey.

### Recency-Based
run_recency_model.py predicts top contacts by selecting the top n nodes who had most recently interacted with the node being predicted for at the time of the survey, where n is the number of recent contacts listed on the survey. If the node being predicted for listed more contacts than the data shows nodes interacted with in the time period from the start of the study to the survey, all nodes interacted with in that time period are selected resulting in a total number predicted less than n.

### Frequency-Based
run_freq_model.py predicts top contacts by selecting the top n nodes who had interacted with the node being predicted for the most time in the time period from the start of the study to the time of the survey, where n is the number of recent contacts listed on the survey. If the node being predicted for listed more contacts than the data shows nodes interacted with in the time period from the start of the study to the survey, all nodes interacted with in that time period are selected resulting in a total number predicted less than n.