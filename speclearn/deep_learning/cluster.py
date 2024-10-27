from speclearn.tools.cache import check_file
import pickle
import numpy as np
from sklearn.cluster import KMeans
from speclearn.tools.constants import CACHE_KMEANS
import os


def cluster_with_kmeans(kmeans_name, data_2d, latent, k, replace=False):
    if not replace and check_file(os.path.join(CACHE_KMEANS, kmeans_name)):
        kmeans = pickle.load(
            open(os.path.join(CACHE_KMEANS, kmeans_name), "rb"))
    else:
        latent_wo_flags = flag_latent(latent, data_2d)

        kmeans = KMeans(n_clusters=k).fit(latent_wo_flags)
        pickle.dump(kmeans, open(os.path.join(
            CACHE_KMEANS, kmeans_name), "wb"))
    return kmeans


def flag_latent(latent, data_2d):
    """Flag latent variables that are not valid (e.g., NaN or less than zeros, both marked with marker of 10)"""
    flags_index = np.argwhere(np.nanmean(np.fabs(data_2d), axis=2) == 10)
    flag_set = set(map(tuple, flags_index))
    latent_index = [i for i, c in enumerate(
        latent) if tuple(c) not in flag_set]
    latent_wo_flags = latent[latent_index]
    return latent_wo_flags


def flag_latent_and_coord(latent, data_2d, coord):
    """Flag latent variables and coordinates that are not valid (e.g., NaN or less than zeros, both marked with marker of 10)"""
    flags_index = np.argwhere((np.nanmean(np.fabs(data_2d), axis=2) == 10) or (
        np.any(np.isnan(data_2d), axis=2)))
    flag_set = set(map(tuple, flags_index))
    latent_index = [i for i, c in enumerate(
        latent) if tuple(c) not in flag_set]
    latent_wo_flags = latent[latent_index]
    coord_wo_flags = coord[latent_index]
    return latent_wo_flags, coord_wo_flags


def flag_latent_and_coord_and_data_2d(latent, data_2d, coord):
    """Flag latent variables, coordinates, and data that are not valid (e.g., NaN or less than zeros, both marked with marker of 10)"""
    flags_index = np.argwhere(
        (np.nanmean(np.fabs(data_2d), axis=2) == 10) | np.any(np.isnan(data_2d), axis=2))
    flag_set = set(map(tuple, flags_index))
    a = data_2d[flags_index]
    latent_index = [i for i, c in enumerate(
        latent) if tuple(c) not in flag_set]

    data = data_2d.reshape(-1, data_2d.shape[-1])
    data_wo_flags = data[latent_index]
    latent_wo_flags = latent[latent_index]
    coord_wo_flags = coord[latent_index]
    return latent_wo_flags, coord_wo_flags, data_wo_flags
