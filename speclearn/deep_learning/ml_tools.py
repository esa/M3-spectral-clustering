import torch
import random
import numpy as np
import json
from speclearn.tools.data_tools import process_data, select_data
from speclearn.tools.constants import *
from speclearn.io.transform.rebin import rebin_from_aoi
from speclearn.tools.data_tools import sub_data
import seaborn as sns
import matplotlib.pyplot as plt
from speclearn.plot.map import plot_map
import math


def get_aoi_dataset(filename='/home/freya/Documents/Code/cache/geojson/aoi_minerals_full.geojson', cuts={}, plot=False, s_1=-12, periods=[], overwrite=False):
    aois = get_aoi_from_geojson(filename)

    aoi_names = [a['name'] for a in aois]
    files = []
    for aoi in aois:
        img_file = rebin_from_aoi(
            aoi, stat='mean', verbose=False, cuts=cuts, periods=periods, overwrite=overwrite)
        _data, data, coord = load_file_ml(img_file, process=False)
        if data.shape[0] > 0:
            files.append(img_file)

    # Split datasets
    X_train, X_test, X_valid = split_datasets(files)

    return X_train, X_valid, X_test

def get_device():
    """
    Get device and check if cuda is available.
    """
    use_cuda = False
    if torch.cuda.is_available():
        dev = "cuda:0"
        use_cuda = True
    else:
        dev = "cpu"
    return dev, use_cuda


device, use_cuda = get_device()


class Config(object):
    """
    Turns a dictionary into a class
    """

    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def load_file_ml(filename, data=None, s_0=0, s_1=-12, process=True, norm=False, as_type: str = '', shuffle=True, crs=False):
    if data is None:
        data = np.load(filename)

    if data.shape[2] > 80:
        data = select_data(data, s_1=s_1)
    if norm:
        data = normalize_data(data)
    if process:
        proc_data, _ = process_data(data, filename, norm=norm, crs=crs)
    else:
        proc_data = data

    dataset, coordinates = ml_ready_data(data=proc_data, as_type=as_type,)

    return proc_data, dataset, coordinates


def ml_ready_data(data, as_type: str = 'tensor'):
    lat = np.linspace(0, data.shape[0]-1, data.shape[0], dtype=int)
    long = np.linspace(0, data.shape[1]-1, data.shape[1], dtype=int)
    lat = np.repeat(lat, data.shape[1])
    long = np.tile(long, data.shape[0])
    coord = list(zip(lat, long))

    new_dim = data.shape[0]*data.shape[1]
    wl_dim = data.shape[2]
    data = data.reshape(new_dim, wl_dim)
    data = data[:, :]
    dataset = []
    coordinates = []
    for d in range(0, len(data[:, 0])):
        if np.any(np.isnan(data[d, :])):
            continue
        else:
            dataset.append(data[d, :])
            coordinates.append(coord[d])
    dataset = np.array(dataset)
    if as_type == 'tensor':
        dataset = get_variable(torch.tensor(dataset, dtype=torch.float))
    return dataset, coordinates


def get_recon_and_latent(model, local_wavelength, ml_data, no_latent=0, eval=True):
    latent = np.full((len(ml_data), no_latent), float('NaN'))
    recon = np.full((len(ml_data), len(local_wavelength)), float('NaN'))

    no_batches = 1000
    i_batch = 0
    while len(ml_data) > i_batch + no_batches:
        batch_data = ml_data[i_batch:(i_batch + no_batches)]
        batch_data = get_variable(torch.tensor(batch_data, dtype=torch.float))
        model.eval()
        _latent, _recon, _, _ = model.forward(batch_data, eval=eval)
        if _latent.shape[1] != no_latent:
            latent[i_batch:(i_batch + no_batches)
                   ] = _latent.cpu().detach().numpy()[:, :no_latent]
        else:
            latent[i_batch:(i_batch + no_batches)
                   ] = _latent.cpu().detach().numpy()
        recon[i_batch:(i_batch + no_batches)] = _recon.cpu().detach().numpy()

        i_batch += no_batches

    batch_data = ml_data[i_batch:]
    batch_data = get_variable(torch.tensor(batch_data, dtype=torch.float))
    _latent, _recon, _, _ = model.forward(batch_data, eval=eval)
    if _latent.shape[1] != no_latent:
        latent[i_batch:(i_batch + no_batches)
               ] = _latent.cpu().detach().numpy()[:, :no_latent]
    else:
        latent[i_batch:(i_batch + no_batches)] = _latent.cpu().detach().numpy()
    recon[i_batch:] = _recon.cpu().detach().numpy()

    return recon, latent


def get_processed_data(local_wavelength, data, ml_data, coord, norm=False, norm_spectra=False, crs=False):
    processed_2d = np.full(
        (data.shape[0], data.shape[1], len(local_wavelength)), float('NaN'))
    processed = np.full((len(ml_data), len(local_wavelength)), float('NaN'))

    no_batches = 60*60
    i_batch = 0
    while len(ml_data) > i_batch + no_batches:
        batch_data = ml_data[i_batch:(i_batch + no_batches)]
        # batch_data, _ = process_data(batch_data, norm=norm, crs=crs)
        batch_data = get_variable(torch.tensor(batch_data, dtype=torch.float))
        processed[i_batch:(i_batch + no_batches)
                  ] = batch_data.cpu().detach().numpy()
        i_batch += no_batches

    batch_data = ml_data[i_batch:]
    # batch_data, _ = process_data(batch_data, norm=norm, crs=crs)
    batch_data = get_variable(torch.tensor(batch_data, dtype=torch.float))
    processed[i_batch:] = batch_data.cpu().detach().numpy()

    for i, (x, y) in enumerate(coord):
        processed_2d[x, y, :] = processed[i, :]

    return processed, processed_2d


def get_poles_aoi(cuts=DEFAULT_CUTS, plot=False, region='south', area='small'):
    if region == 'south':
        if area == 'small':
            aoi = {
                'name': 'loc_-180_180_-90_-80',  # 'south_dbin_005_all',#south_lat90_80_dbin_005
                'lat_range': [-90.0, -80.0],
                'long_range': [-180.0, 180.0],
                'd_lat': 0.05,
                'd_long': 0.05
            }
        else:
            aoi = {
                'name': 'loc_-180_180_-90_-60',  # 'south_dbin_005_all',#south_lat90_80_dbin_005
                'lat_range': [-90.0, -60.0],
                'long_range': [-180.0, 180.0],
                'd_lat': 0.05,
                'd_long': 0.05
            }
    else:
        if area == 'small':
            aoi = {
                'name': 'loc_-180_180_80_90',  # 'south_dbin_005_all',#south_lat90_80_dbin_005
                'lat_range': [80, 90],
                'long_range': [-180.0, 180.0],
                'd_lat': 0.05,
                'd_long': 0.05
            }
        else:
            aoi = {
                'name': 'loc_-180_180_60_90',  # 'south_dbin_005_all',#south_lat90_80_dbin_005
                'lat_range': [60, 90],
                'long_range': [-180.0, 180.0],
                'd_lat': 0.05,
                'd_long': 0.05
            }
    return aoi


def random_sample_and_remove(data, n):
    """
    Randomly samples n elements from a list and removes them from the original list.

    Args:
      data: The list to sample from.
      n: The number of elements to sample.

    Returns:
      A new list containing n randomly sampled elements, which are removed from the input list.
    """
    sampled_indices = random.sample(range(len(data)), n)
    sampled_elements = [data[i] for i in sampled_indices]
    # Remove sampled elements in reverse order to avoid index issues
    for i in sorted(sampled_indices, reverse=True):
        del data[i]
    return sampled_elements, data


def get_one_degree_aoi_list():
    aoi_list = []
    for lat_start in range(-90, 90, 1):  # 1 degree latitude bins
        for long_start in range(-180, 180, 1):  # 1 degree longitude bins
            lat_end = lat_start + 1
            long_end = long_start + 1
            name = f'loc_{long_start}_{long_end}_{lat_start}_{lat_end}'

            aoi_list.append(
                {
                    'name': name,
                    'lat_range': [lat_start, lat_end],
                    'long_range': [long_start, long_end],
                    'd_lat': 0.05,
                    'd_long': 0.05
                }
            )
    return aoi_list


def divide_data(aoi_list, limit=None):
    random.shuffle(aoi_list)
    if limit is not None:
        aoi_list = aoi_list[:limit]
    total_length = len(aoi_list)
    train_length = math.floor(0.60 * total_length)
    val_length = math.floor(0.20 * total_length)
    test_length = total_length - train_length - \
        val_length  # Ensure the lengths add up

    training_data, aoi_list = random_sample_and_remove(aoi_list, train_length)
    validation_data, aoi_list = random_sample_and_remove(aoi_list, val_length)
    test_data, aoi_list = random_sample_and_remove(aoi_list, test_length)

    print('No. training files:', len(training_data))
    print('No. validation files:', len(validation_data))
    print('No. test files:', len(test_data))
    return training_data, validation_data, test_data


def get_data_files_from_aoi(aoi_list, periods=[]):
    files = []
    for aoi in aoi_list:
        img_file = rebin_from_aoi(
            aoi, stat='mean', verbose=False, periods=periods)
        _data, data, coord = load_file_ml(img_file, process=False)
        if data.shape[0] > 0:
            files.append(img_file)
    return files


def split_datasets(data_files):
    """
    Split dataset into 80 % for training, 10 % for validation, 10 % for testing
    """

    X_test = []
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/TiO2_mean.npy')
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/Orthopyroxene_mean.npy')
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/Olivine_mean.npy')
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/Olivine_3_mean.npy')
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/FeO_2_mean.npy')
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/Random_18_mean.npy')
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/Random_35_mean.npy')
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/Random_61_mean.npy')
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/Random_13_mean.npy')
    X_test.append(
        '/media/freya/rebin/M3/pickles/nlong7200_nlat3600/sun_zenith_90_sensor_zenith_25_obs_phase_angle_180/spec/Random_2_mean.npy')
    for f in X_test:
        if f in data_files:
            data_files.remove(f)

    # random.shuffle(data_files) # FIXME enable eventually
    train_index = int(len(data_files) * 0.8)
    valid_index = train_index + int(len(data_files) * 0.2)

    X_train = data_files[0:train_index]
    # X_test = data_files[train_index:valid_index]
    X_valid = data_files[train_index:]

    return X_train, X_test, X_valid


def get_aoi_from_geojson(geojson_file, d_lat=0.05, d_long=0.05):
    with open(geojson_file) as f:
        data = json.load(f)

    name = data['features'][0]['properties']['label']
    coordinates = data['features'][0]['geometry']['coordinates']
    lat_range = [coordinates[0][1][1], coordinates[0][3][1]]
    long_range = [coordinates[0][1][0], coordinates[0][3][0]]
    aois = []
    for i in range(0, len(data['features'])):
        name = data['features'][i]['properties']['label']
        coordinates = data['features'][i]['geometry']['coordinates']
        lat_range = [coordinates[0][1][1], coordinates[0][3][1]]
        long_range = [coordinates[0][1][0], coordinates[0][3][0]]
        if long_range[0] > long_range[1]:
            long_range = np.flip(long_range)
        if lat_range[0] > lat_range[1]:
            lat_range = np.flip(lat_range)

        aoi = {
            'name': name,
            'lat_range': lat_range,
            'long_range': long_range,
            'd_lat': d_lat,
            'd_long': d_long
        }
        aois.append(aoi)
    return aois
