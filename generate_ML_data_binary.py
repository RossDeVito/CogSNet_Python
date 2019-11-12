import os
import pickle
import time
from itertools import combinations

import numpy as np
import pandas as pd

from run_cogsnet import get_forget_intensity, get_signals


def get_recency(events, survey_time, earliest_timestamp=1312617635):
    """ Returns time delta between survey time and most recent event.

    If there are no events before or at survey time, returns time between
    earliest timestamp and survey time. earliest_timestamp should be the 
    earliest timestamp in the whole data set, not just the events in events.
    """
    events_at_time = events[events <= survey_time]
    
    if len(events_at_time) == 0:
        return survey_time - earliest_timestamp

    return survey_time - max(events_at_time)


def get_volume(events, survey_time):
    """ Returns count of events before survey time """
    return len(events[events <= survey_time])


def get_volume_n_days_before(events, survey_time, n_days):
    return len(events[(events > (survey_time - n_days * 86400))
                        & (events <= survey_time)])


def get_hawkes(event_times, observation_time, beta):
    times_before_obs = event_times[event_times <= observation_time]
    time_deltas = observation_time - times_before_obs

    return np.sum(beta * np.exp(-beta * time_deltas))


if __name__ == "__main__":
    with open(os.path.join("data", "edge_dict.pkl"), 'rb') as pkl:
        edge_dict = pickle.load(pkl)

    with open(os.path.join("data", "interaction_dict.pkl"), 'rb') as pkl:
        interaction_dict = pickle.load(pkl)

    with open(os.path.join("data", "survey_dict.pkl"), 'rb') as pkl:
        survey_dict = pickle.load(pkl)

    L = 21
    mu = 0.2
    theta = 0.166667
    forget_type = 'exp'
    forget_intensity = get_forget_intensity(L, mu, theta, forget_type)

    # desc holds data about source of row's comparison
    desc_col_names = ['respondant_id', 'id_1', 'id_2', 'survey_time']
    desc_rows = []

    # id_1 and id_2's feature vectors joined followed by label.
    #   label is 1 if id_1 outranked id_2, else 0
    data_col_names = ['id_1_recency', 'id_1_volume', 'id_1_vol_3_week', 
                        'id_1_CogSNet', 'id_1_hawkes', 
                        'id_2_recency', 'id_2_volume', 'id_2_vol_3_week', 
                        'id_2_CogSNet', 'id_2_hawkes',
                        'label']
    data_rows = []
    
    for respondant_id, surveys in survey_dict.items():
        if (respondant_id not in edge_dict.keys()):
            continue

        for survey_time, survey in surveys.items():
            for rank_1, rank_2 in combinations(survey.keys(), 2):
                r1_id = survey[rank_1]
                r2_id = survey[rank_2]

                if (r1_id in interaction_dict[respondant_id].keys()):
                    r1_events = interaction_dict[respondant_id][r1_id][:, 2]
                else:
                    r1_events = np.empty(0)
                if (r2_id in interaction_dict[respondant_id].keys()):
                    r2_events = interaction_dict[respondant_id][r2_id][:, 2]
                else:
                    r2_events = np.empty(0)

                r1_recency = get_recency(r1_events, survey_time)
                r2_recency = get_recency(r2_events, survey_time)

                r1_volume = get_volume(r1_events, survey_time)
                r2_volume = get_volume(r2_events, survey_time)

                r1_vol_21 = get_volume_n_days_before(r1_events, survey_time, 21)
                r2_vol_21 = get_volume_n_days_before(r2_events, survey_time, 21)

                if len(r1_events > 0):
                    r1_cogs = get_signals(r1_events, [survey_time], mu, theta,
                                            forget_type, forget_intensity)[0]
                else:
                    r1_cogs = 0.0
                if len(r2_events > 0):
                    r2_cogs = get_signals(r2_events, [survey_time], mu, theta,
                                          forget_type, forget_intensity)[0]
                else:
                    r2_cogs = 0.0

                r1_hawkes = get_hawkes(r1_events, survey_time, 1.727784e-07)
                r2_hawkes = get_hawkes(r2_events, survey_time, 1.727784e-07)

                if rank_1 < rank_2:
                    label_1 = 1
                    label_2 = 0
                else:
                    label_1 = 0
                    label_2 = 1

                # add row where r1_id is id_1
                desc_rows.append([respondant_id, r1_id, r2_id, survey_time])
                data_rows.append(
                    [r1_recency, r1_volume, r1_vol_21, r1_cogs, r1_hawkes,
                     r2_recency, r2_volume, r2_vol_21, r2_cogs, r2_hawkes,
                     label_1])

                # add row where r2_id is id_1
                desc_rows.append([respondant_id, r2_id, r1_id, survey_time])
                data_rows.append(
                    [r2_recency, r2_volume, r2_vol_21, r2_cogs, r2_hawkes,
                     r1_recency, r1_volume, r1_vol_21, r1_cogs, r1_hawkes,
                     label_2])
                
                
    desc_df = pd.DataFrame(desc_rows, columns=desc_col_names)
    desc_df.to_pickle("data/pairwise_binary_desc.pkl")
    data_df = pd.DataFrame(data_rows, columns=data_col_names)
    data_df.to_pickle("data/pairwise_binary_data.pkl")
