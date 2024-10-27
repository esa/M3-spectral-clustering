import datetime
import glob
import os
import warnings
import pandas as pd

import numpy as np
from scipy import stats
from speclearn.io.data.M3_image import M3_image
from speclearn.tools.constants import *

from spectral import *
from speclearn.tools.constants import *
from speclearn.tools.data_tools import _in_OP1A, _in_OP1B, _in_OP2A, _in_OP2B, _in_OP2C, _in_OP2C1


def select_images_in_period(image_files=[], periods=[]):
    images_in_period = []
    for f in image_files:
        date = M3_image(f).get_info().date
        dt = datetime.datetime.strptime(str(date), '%Y%m%d')

        if ('OP1A' in periods) and (_in_OP1A(dt)):
            images_in_period.append(f)
        if ('OP1B' in periods) and (_in_OP1B(dt)):
            images_in_period.append(f)
        if ('OP2A' in periods) and (_in_OP2A(dt)):
            images_in_period.append(f)
        if ('OP2B' in periods) and (_in_OP2B(dt)):
            images_in_period.append(f)
        if ('OP2C' in periods) and (_in_OP2C(dt)):
            images_in_period.append(f)
        if ('OP2_C1' in periods) and (_in_OP2C1(dt)):
            images_in_period.append(f)
    return images_in_period


def get_images_in_range(long_range, lat_range, periods=[], verbose=False):
    if len(periods) > 0:
        period_name = '_'+'_'.join(periods)
    else:
        period_name = ''
    filename = f'{M3_PICKLE_DATA_DIR}/image_files/loc_{long_range[0]}-{
        long_range[1]}_{lat_range[0]}-{lat_range[1]}{period_name}.npy'

    if os.path.isfile(filename):
        image_files = np.load(filename, allow_pickle=True)
        image_files = remove_bad_files(image_files)
    else:
        image_files = search_images_in_range([long_range[0], long_range[1]], [
                                             lat_range[0], lat_range[1]])
        image_files = remove_bad_files(image_files)
        np.save(filename, np.array(image_files))
    if verbose:
        print(f'Found {len(image_files)} files in range')

    if len(periods) > 0:
        image_files = select_images_in_period(image_files, periods=periods)
    return image_files


def remove_zeros_and_ones(img, return_zeros=False):
    img_rm = img.copy()
    mean = np.nanmean(img_rm, axis=2)
    std = np.nanstd(img_rm, axis=2)

    img_zeros = None
    if return_zeros:
        img_zeros = img_rm.copy()
        img_zeros[(mean > 0.01)] = float('NaN')
        img_zeros[(std > 0.01)] = float('NaN')

    img_rm[(mean < 0.01)] = float('NaN')
    img_rm[(std < 0.01)] = float('NaN')

    # remove spectra where any is larger than 1 (with some wiggle room)
    img_rm[(np.nanmax(img_rm, axis=2) > 1.2)] = float('NaN')
    return img_rm, img_zeros


def remove_outliers_spectral_line(img, threshold=5, verbose=False, return_outliers=False):
    """
    Remove outliers in the spectral dimension of an image (i.e., for each spectral line img[i,j,:]).
    If an outlier is found in a spectral line, the entire line is set to NaN.
    Optimized for speed using NumPy's vectorized operations.

    Args:
      img: A 3D numpy array representing the image, where the spectral dimension is the third dimension.
      threshold: The z-score threshold for outlier detection.
      verbose: A boolean indicating whether to print the number of outliers found.

    Returns:
      A 3D numpy array with the outlier spectral lines replaced by NaN.
    """

    # Calculate z-scores for all spectral lines at once
    z = np.abs(stats.zscore(img, nan_policy='omit', axis=2))

    # Find outliers based on the threshold
    outliers = z > threshold

    # Check if any outlier exists in each spectral line
    outlier_lines = np.any(outliers, axis=2)

    # Warn if any spectral line has outliers
    if np.any(outlier_lines):
        warnings.warn('Outliers found in some spectral lines.')

    if verbose:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if outlier_lines[i, j]:
                    print(f'Found outliers in spectral line [{i}, {j}, :]')

    # Set entire spectral lines with outliers to NaN
    img_rm = img.copy()
    img_outliers = None
    if return_outliers:
        img_outliers = np.full_like(img_rm, float('NaN'))
        img_outliers[outlier_lines, :] = img_rm[outlier_lines, :]

    img_rm[outlier_lines, :] = float('NaN')

    return img_rm, img_outliers


def remove_outliers(img, refl=False, threshold=10, verbose=False):
    """
    Remove flags before using this function.
    """
    # find outliers
    print(img.img_hdr)
    if refl:
        z = np.array(
            np.abs(stats.zscore(img.reflectance[:, :], nan_policy='omit', axis=None)))
        no_outliers = len(z[z > threshold])
        if no_outliers / len(img.reflectance[:, :].flatten()) > 0.1:
            warnings.warn('More than 10 percent are outliers.')

        if verbose:

            print(f'Found {no_outliers} outliers')
        img.reflectance[:, :][z > threshold] = float('NaN')
    else:
        for wl in range(0, len(GLOBAL_WAVELENGTH)):
            # find outliers
            z = np.array(
                np.abs(stats.zscore(img.data[:, :, wl], nan_policy='omit', axis=None)))
            no_outliers = len(z[z > threshold])
            if no_outliers / len(img.data[:, :, wl].flatten()) > 0.1:
                warnings.warn('More than 10 percent are outliers.')
            if verbose:
                print(f'Found {no_outliers} outliers')

            img.data[:, :, wl][z > threshold] = float('NaN')
    return img


def remove_bad_files(files):
    good_files = []
    bad_files = ['M3G20090207T003331_V01_RFL', 'M3G20081118T223204_V01_RFL']
    for file in files:
        name = file.split('.')[-2].split('/')[-1]
        if name in bad_files:
            continue
        else:
            good_files.append(file)
    return good_files


def data_generator(image_files=[], verbose=False):
    if verbose:
        print('-- data_generator')
    for file in image_files:
        if verbose:
            print(f'data_generator: file {file}')

        img = M3_image(file, level=2)
        img.get_info()

        yield img


def fill_and_select(img, long_range=[], lat_range=[], bands=[], refl=False, obs=False, cuts=DEFAULT_CUTS, proj=''):
    # TODO: pass refl
    img.fill()
    if obs:
        img.fill_observation()
    if len(proj) > 0:
        img.fill_location(proj=proj)
    if not refl:
        img.load_data()
        img = remove_outliers(img)
    if (len(long_range) > 0) & (len(lat_range) > 0):
        select = (img.longitude >= long_range[0]) & (img.longitude <= long_range[1]) \
            & (img.latitude >= lat_range[0]) & (img.latitude <= lat_range[1])
        img = img[select]
    if len(bands) > 0:
        img.data = img.data[:, bands]
    # img = apply_cuts(img, cuts)
    return img


def apply_cuts(img, cuts=DEFAULT_CUTS):
    if len(cuts) > 0 and img.latitude is not None:
        select = (img.sun_zenith <= cuts['sun_zenith']) \
            & (img.sensor_zenith <= cuts['sensor_zenith']) \
            & (img.obs_phase_angle <= cuts['obs_phase_angle'])
        img = img[select]
    return img


def search_images_in_range(long_range, lat_range, verbose=False):
    if verbose:
        print('speclearn: Look for files in range')

    image_files = []
    for file in L2_FILES:
        m3_img = M3_image(file, level=2)
        try:
            m3_img.fill_location(level=m3_img.level)
            if np.any((m3_img.longitude >= long_range[0])
                      & (m3_img.longitude <= long_range[1])
                      & (m3_img.latitude >= lat_range[0])
                      & (m3_img.latitude <= lat_range[1])):
                image_files.append(file)
            else:
                file_split = m3_img.img_hdr.split('/')[:]
                data_dir = '/'.join(file_split[:-6])
                file_path = data_dir + '/' + \
                    '/'.join(['CH1M3_0003', 'DATA',
                             m3_img.date_interval, m3_img.date[0:6], 'L1B'])
                headers = glob.glob(os.path.join(
                    file_path, m3_img.data_file + '*LOC.HDR'))
                images = glob.glob(os.path.join(
                    file_path, m3_img.data_file + '*LOC.IMG'))
                header_obs = glob.glob(os.path.join(
                    file_path, m3_img.data_file + '*OBS.HDR'))
                images_obs = glob.glob(os.path.join(
                    file_path, m3_img.data_file + '*OBS.IMG'))
        except:

            print('Could not fill location.')
            print(file)
            pass
        finally:
            continue
    print(f'{len(image_files)} files in range')
    return image_files


def find_data_file(name):
    for l in L2_FILES:
        if name in l:
            return (l)


def get_image_files_from_label(label_files):
    image_files = []
    for f in label_files:
        hdr = f.split('_')[0] + '_V01_RFL.HDR'
        new_file = find_data_file(hdr)
        image_files.append(new_file)
    image_files = remove_bad_files(image_files)
    return image_files


def load_file(file):
    loaded_file = np.load(file, allow_pickle=True)
    binned = loaded_file['data']
    bins = loaded_file['bins'].tolist()
    return binned, bins


def rebin_in_range(rebin_file, aoi):
    data, bins = load_file(rebin_file)
    long = bins['long_center']
    lat = bins['lat_center']

    df_2d = pd.DataFrame(
        data=data[:, :].T, index=bins['lat_center'], columns=bins['long_center'])
    df_2d = df_2d.sort_index(ascending=False)
    df_2d = df_2d.reindex(sorted(df_2d.columns), axis=1)
    select_lat = (df_2d.index >= aoi['lat_range'][0]) & (
        df_2d.index <= aoi['lat_range'][1])
    select_long = (df_2d.columns >= aoi['long_range'][0]) & (
        df_2d.columns <= aoi['long_range'][1])
    df_2d = df_2d[select_lat].T[select_long]
    return df_2d
