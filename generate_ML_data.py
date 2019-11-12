import os
import pickle
import time
from itertools import product

import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import rbo

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_pca(X):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_r1 = pca.fit_transform(X)

    fig, ax = plt.subplots()
    for i in range(len(X_r1)):
        plt.scatter(X_r1[i][0], X_r1[i][1], c='r')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('PCA on TARGET')
    plt.show()


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def rbo_score(list1, list2):
    return rbo.RankingSimilarity(list1, list2).rbo()


def freq_features(events, surveys, day_bins=21):
    second_bins = day_bins * 86400
    freq = {}
    for survey_time in surveys.keys():
        start_time = survey_time - second_bins
        if start_time < 0:
            start_time = 0
        canidate_event_counts = {}
        for k, v in events.items():
            if len(v[np.logical_and(v[:, 2] >= start_time, v[:, 2] <= survey_time)]) > 0:
                canidate_event_counts[k] = len(
                    v[np.logical_and(v[:, 2] >= start_time, v[:, 2] <= survey_time)])

        freq[survey_time] = canidate_event_counts
    return freq


def vol_features(events, surveys):
    vol = {}
    for survey_time in surveys.keys():
        canidate_event_counts = {k: len(v[np.asarray(v[:, 2] <= survey_time)]) for k, v in events.items(
        ) if len(v[np.asarray(v[:, 2] <= survey_time)] > 0)}
        vol[survey_time] = canidate_event_counts
    return vol


def recency_features(events, surveys):
    rec = {}
    for survey_time in surveys.keys():
        canidate_most_recent = {k: max(v[np.asarray(v[:, 2] <= survey_time)][:, 2])
                                for k, v in events.items() if len(v[np.asarray(v[:, 2] <= survey_time)] > 0)}
        rec[survey_time] = canidate_most_recent
    return rec


def get_forget_intensity(lifetime, mu, theta, forget_type):
    if forget_type == 'pow':
        return np.log(mu / theta) / np.log(lifetime)
    elif forget_type == 'exp':
        return np.log(mu / theta) / lifetime


def forget_func(forget_type, time_delta, forget_intensity):
    if forget_type == 'pow':
        return max(1, time_delta) ** (-1 * forget_intensity)
    elif forget_type == 'exp':
        return np.e ** (-1 * forget_intensity * time_delta)


def hawkes_features(event_times, observation_times, beta):
	ret_values = []

	event_times = np.asarray(event_times)

	for obs_time in observation_times:
		times_before_obs = event_times[event_times < obs_time]
		time_deltas = obs_time - times_before_obs
		ret_values.append(
			np.sum(beta * np.exp(-beta * time_deltas))
		)

	return np.array(ret_values)


def cogsnet_features(start_times, observation_times, mu, theta, forget_type, forget_intensity):
    start_times = sorted(start_times)
    observation_times = sorted(observation_times)

    ret_values = []

    current_signal = 0
    obs_ind = 0
    total_obs = len(observation_times)

    if len(start_times) > 0:
        while start_times[0] > observation_times[obs_ind]:
            ret_values.append(current_signal)
            obs_ind += 1
            if obs_ind >= total_obs:
                return ret_values

        current_signal = mu

    if obs_ind >= total_obs:
        return ret_values

    for i in range(1, len(start_times)):
        while start_times[i] > observation_times[obs_ind]:
            val_at_obs = current_signal * \
                forget_func(
                    forget_type, (observation_times[obs_ind] - start_times[i-1]) / 3600, forget_intensity)

            if val_at_obs < theta:
                ret_values.append(0)
            else:
                ret_values.append(val_at_obs)

            obs_ind += 1

            if obs_ind >= total_obs:
                break

        if obs_ind >= total_obs:
            break

        decayed_signal = current_signal * \
            forget_func(
                forget_type, (start_times[i] - start_times[i-1]) / 3600, forget_intensity)
        if decayed_signal < theta:
            decayed_signal = 0
        current_signal = mu + decayed_signal * (1 - mu)

    while obs_ind < total_obs:
        val_at_obs = current_signal * \
            forget_func(
                forget_type, (observation_times[obs_ind] - start_times[-1]) / 3600, forget_intensity)

        if val_at_obs < theta:
            ret_values.append(0)
        else:
            ret_values.append(val_at_obs)

        obs_ind += 1

    return ret_values


def get_rank(survey_data, survey_time, edge):
    rank_data = survey_data[survey_time]
    for rank in rank_data:
        if rank_data[rank] == edge:
            return int(rank)
    else:
        return -1


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

    data = {}
    labels = {}
    for participant_id in survey_dict.keys():
        if (participant_id in edge_dict.keys()):
            data[participant_id] = {}
            labels[participant_id] = {}

            events = interaction_dict[participant_id]
            surveys = survey_dict[participant_id]

            forget_intensity = get_forget_intensity(L, mu, theta, forget_type)

            node_ids = np.asarray(list(events.keys()))
            survey_times = list(surveys.keys())

            cog = np.asarray([cogsnet_features(events_mat[:, 2], survey_times, mu,
                                               theta, forget_type, forget_intensity) for events_mat in events.values()])
            rec = recency_features(events, surveys)
            vol = vol_features(events, surveys)
            freq = freq_features(events, surveys, day_bins=L)

            edge_list = list(events.keys())
            survey_list = list(surveys.keys())
            for j in range(len(edge_list)):
                X = []
                y = []
                edge = edge_list[j]

                for i in range(len(survey_list)):
                    survey = survey_list[i]
                    cog_val = cog[j, i]

                    rec_val = rec[survey]
                    if edge in rec_val:
                        rec_val = rec[survey][edge]
                    else:
                        rec_val = 0

                    vol_val = vol[survey]
                    if edge in vol_val:
                        vol_val = vol[survey][edge]
                    else:
                        vol_val = 0

                    freq_val = freq[survey]
                    if edge in freq_val:
                        freq_val = freq[survey][edge]
                    else:
                        freq_val = 0

                    X.append([cog_val, rec_val, vol_val, freq_val])
                    y.append(get_rank(surveys, survey, edge))

                X = np.array(X)
                y = np.array(y)

                data[participant_id][edge] = X
                labels[participant_id][edge] = y

# data/ml_data -> data | labels

# data -> participant_id : edge : X
# X -> (row, col) -> row : survey number (0 to N) | col : cog_val, rec_val, vol_val, freq_val

# NOTE: X has no scaling. Do that after loading in.

# labels -> participant_id : edge : y
# y -> (row, ) -> row : survey number (0 to N)

res = {'data': data, 'labels': labels}

ofile = "data/ml_data.pkl"
with open(ofile, 'wb') as pkl:
    pickle.dump(res, pkl, protocol=pickle.HIGHEST_PROTOCOL)
