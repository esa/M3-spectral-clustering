import os

import numpy as np
import torch

from speclearn.deep_learning.ml_tools import (get_recon_and_latent,
                                              get_variable, load_file_ml)
from speclearn.io.transform.rebin import rebin_from_aoi
from speclearn.tools.cache import check_file, get_rebin_path
from speclearn.tools.constants import CACHE_PREDICT, M3_PICKLE_AOI_DIR, M3_PICKLE_DATA_DIR
from speclearn.tools.data_tools import (mark_zeros, process_data, select_data,
                                        select_wavelength)


def predict_data(filename, model, data=None, norm=True, crs=True, process=True, no_latent=5, s_0=0, s_1=-12):
    local_wavelength = select_wavelength(s_0=s_0, s_1=s_1)
    if filename is not None:
        filename = os.path.join(M3_PICKLE_AOI_DIR, filename)
    data, ml_data, coord = load_file_ml(
        filename, data=data, process=process, norm=norm, s_1=s_1, as_type='', crs=crs)

    if len(ml_data) == 0:
        return None, None, None, None, None
    recon, latent = get_recon_and_latent(
        model, local_wavelength, ml_data, no_latent=no_latent)
    return recon, latent, data, coord, ml_data


def get_names(crs, norm, periods):
    crs_name = ''
    norm_name = ''

    if norm:
        norm_name = '_norm'
    if crs:
        crs_name = '_crs'
        norm_name = ''
    if len(periods) > 0:
        period_name = '_'+periods[0]
    else:
        period_name = ''
    return crs_name, norm_name, period_name


def make_data_input_file(pickle, crs, norm):
    data_raw = np.load(pickle)[:, :, :-12]
    data, _ = process_data(data_raw, filename=pickle,
                           exclude_crs=True, marker=float('NaN'), crs=crs, norm=norm)
    return data


def predict_full_map(aois, model, model_name, no_latent=5, crs=True, periods=[], norm=True):
    for aoi in aois:
        print(aoi)

        pickle = rebin_from_aoi(aoi, refl=False, periods=periods)

        latitude_range = f'{int(aoi['lat_range'][0])}_{
            int(aoi['lat_range'][1])}'
        longitude_range = f'{int(aoi["long_range"][0])}_{
            int(aoi["long_range"][1])}'

        crs_name, norm_name, period_name = get_names(
            crs=crs, norm=norm, periods=periods)

        if not check_file(f'{CACHE_PREDICT}/{model_name}_latent_{longitude_range}_{latitude_range}{period_name}.npy'):
            print(
                f'{CACHE_PREDICT}/{model_name}_recon_{longitude_range}_{latitude_range}{period_name}.npy')
            data_input_file = f'{CACHE_PREDICT}/data_{longitude_range}_{
                latitude_range}{period_name}{norm_name}{crs_name}.npy'

            if not check_file(data_input_file):
                data_input = make_data_input_file(
                    pickle=pickle, crs=crs, norm=norm)
                np.save(data_input_file, data_input)

            else:
                print('data_input_file exists')
                data_input = np.load(data_input_file)
            data_input[np.fabs(data_input) == 10] = float('NaN')

            recon, latent, _data, coord, ml_data = predict_data(filename=None,
                                                                no_latent=no_latent,
                                                                model=model, crs=crs, data=data_input, process=False, norm=False)

            if recon is not None:
                np.save(
                    f'{CACHE_PREDICT}/{model_name}_recon_{longitude_range}_{latitude_range}{period_name}', recon)
                np.save(
                    f'{CACHE_PREDICT}/{model_name}_latent_{longitude_range}_{latitude_range}{period_name}', latent)
                np.save(
                    f'{CACHE_PREDICT}/coord_{longitude_range}_{latitude_range}{period_name}{norm_name}{crs_name}', coord)
        else:
            print(
                f'{CACHE_PREDICT}/{model_name}_recon_{longitude_range}_{latitude_range}.npy exists')


def process_full_map(aois, crs=True, periods=[], norm=True):
    for aoi in aois:
        print(aoi)

        mid_pickle = rebin_from_aoi(aoi, refl=False, periods=periods)

        lat_range = aoi['lat_range']
        latitude_range = f'{int(lat_range[0])}_{int(lat_range[1])}'

        crs_name, norm_name, period_name = get_names(
            crs=crs, norm=norm, periods=periods)
        data_input_file = f'{
            CACHE_PREDICT}/data_{latitude_range}{period_name}{norm_name}{crs_name}.npy'

        print(data_input_file)
        if not check_file(data_input_file):

            data_input = make_data_input_file(
                mid_pickle=mid_pickle, crs=crs, norm=norm)
            np.save(data_input_file, data_input)

        else:
            data_input = np.load(data_input_file)
        data_input[np.fabs(data_input) == 10] = float('NaN')


def read_area(aois, model_name, crs, norm, full, area):
    if full:
        data_2d_crs, coord_crs, latent_crs = get_data(
            aois, model_name, crs=crs, norm=norm)  # -2:2
    else:
        data_2d_crs, coord_crs, latent_crs = get_data(
            aois[2:-2], model_name, crs=crs, norm=norm)  # -2:2

    if area == 'small':
        data_2d_s_crs, coord_s_crs, latent_s_crs = get_data(
            aois[0:1], model_name, crs=crs, norm=norm)
        data_2d_n_crs, coord_n_crs, latent_n_crs = get_data(
            aois[-1:], model_name, crs=crs, norm=norm)
    else:
        data_2d_s_crs, coord_s_crs, latent_s_crs = get_data(
            aois[0:3], model_name, crs=crs, norm=norm)
        data_2d_n_crs, coord_n_crs, latent_n_crs = get_data(
            aois[-3:], model_name, crs=crs, norm=norm)
    return data_2d_crs, coord_crs, latent_crs, data_2d_s_crs, coord_s_crs, latent_s_crs, data_2d_n_crs, coord_n_crs, latent_n_crs


def get_full_data(aois, crs, norm, periods=[], suffix=''):
    full_data = []
    full_coord = []
    i = 0
    crs_name, norm_name, period_name = get_names(
        crs=crs, norm=norm, periods=periods)
    full_file_name = f"{CACHE_PREDICT}/data_-180_180_-90_90.npy"

    if not os.path.exists(full_file_name):
        for aoi in aois:
            lat_range = aoi['lat_range']
            long_range = aoi['long_range']
            latitude_range = f'{int(lat_range[0])}_{int(lat_range[1])}'
            longitude_range = f'{int(long_range[0])}_{int(long_range[1])}'
            data_input_file = f'{
                CACHE_PREDICT}/data_{longitude_range}_{latitude_range}{period_name}{norm_name}{crs_name}.npy'

            data = np.load(data_input_file)
            full_data.append(data)

            # coord = np.load(
            #    f'{CACHE_PREDICT}/coord_{longitude_range}_{latitude_range}{period_name}.npy')
            # coord[:, 0] += i

            i += 400

            # full_coord.append(coord)
        full_data = np.concatenate(full_data)
    else:
        full_data = np.load(full_file_name)
    return full_data  # , full_coord


def get_coord(aois, crs=False, norm=False, periods=[]):
    full_data = []
    full_coord = []

    i = 0
    crs_name, norm_name, period_name = get_names(
        crs=crs, norm=norm, periods=periods)

    for aoi in aois:
        lat_range = aoi['lat_range']
        long_range = aoi['long_range']
        latitude_range = f'{int(lat_range[0])}_{int(lat_range[1])}'
        longitude_range = f'{int(long_range[0])}_{int(long_range[1])}'
        data_input_file = f'{
            CACHE_PREDICT}/data_{longitude_range}_{latitude_range}{period_name}{norm_name}{crs_name}.npy'

        coord = np.load(
            f'{CACHE_PREDICT}/coord_{longitude_range}_{latitude_range}{period_name}.npy')
        coord[:, 0] += i

        i += 400

        full_coord.append(coord)
    full_coord = np.concatenate(full_coord)
    return full_coord


def get_data(aois, model_name, periods=[], crs=False, norm=False):
    full_latent = []
    full_recon = []
    full_data = []
    full_coord = []

    i = 0
    crs_name, norm_name, period_name = get_names(
        crs=crs, norm=norm, periods=periods)

    full_latent = get_latent(aois, model_name, periods=periods)
    print('get latent')
    full_data = get_full_data(aois, crs, norm, periods=periods)
    full_recon = get_full_recon(aois, model_name, periods=periods)
    full_coord = get_coord(aois, crs, norm, periods=periods)
    return full_data, full_coord, full_latent, full_recon


def get_latent(aois, model_name, periods=[]):
    full_latent = []

    i = 0
    crs_name, norm_name, period_name = get_names(
        crs=False, norm=False, periods=periods)

    for aoi in aois:
        lat_range = aoi['lat_range']
        long_range = aoi['long_range']
        latitude_range = f'{int(lat_range[0])}_{int(lat_range[1])}'
        longitude_range = f'{int(long_range[0])}_{int(long_range[1])}'

        latent = np.load(
            f'{CACHE_PREDICT}/{model_name}_latent_{longitude_range}_{latitude_range}{period_name}.npy')

        full_latent.append(latent)

    full_latent = np.concatenate(full_latent)

    return full_latent


def get_full_recon(aois, model_name, periods=[], crs=True, norm=True):
    full_recon = []

    crs_name, norm_name, period_name = get_names(
        crs=crs, norm=norm, periods=periods)

    for aoi in aois:
        lat_range = aoi['lat_range']
        long_range = aoi['long_range']
        latitude_range = f'{int(lat_range[0])}_{int(lat_range[1])}'
        longitude_range = f'{int(long_range[0])}_{int(long_range[1])}'

        recon = np.load(
            f'{CACHE_PREDICT}/{model_name}_recon_{longitude_range}_{latitude_range}{period_name}.npy')
        full_recon.append(recon)

    full_recon = np.concatenate(full_recon)

    return full_recon
