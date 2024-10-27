
from speclearn.tools.cache import check_file
import warnings

import numpy as np
import pysptools.spectro as spectro
from scipy.signal import savgol_filter

from speclearn.tools.constants import (GLOBAL_WAVELENGTH, OP1A_END, OP1A_START,
                                       OP1B_END, OP1B_START, OP2A_END,
                                       OP2A_START, OP2B_END, OP2B_START,
                                       OP2C_END, OP2C_START, OP2C1_END, OP2C1_START, normalize_data)
from speclearn.tools.hull import extract_crs


def mark_zeros(data, marker=float('NaN')):
    """
    Mark zero pixels in data with marker.
    """

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.nanmean(data[i, j, :]) < 0.05:
                data[i, j, :] = marker
    return data


def _in_OP1A(dt):
    if (dt >= OP1A_START) and (dt <= OP1A_END):
        return True
    return False


def _in_OP1B(dt):
    if (dt >= OP1B_START) and (dt <= OP1B_END):
        return True
    return False


def _in_OP2A(dt):
    if (dt >= OP2A_START) and (dt <= OP2A_END):
        return True
    return False


def _in_OP2B(dt):
    if (dt >= OP2B_START) and (dt <= OP2B_END):
        return True
    return False


def _in_OP2C(dt):
    if (dt >= OP2C_START) and (dt <= OP2C_END):
        return True
    return False


def _in_OP2C1(dt):
    if (dt >= OP2C1_START) and (dt <= OP2C1_END):
        return True
    return False


def _in_OP2C1(dt):
    if (dt >= OP2C1_START) and (dt <= OP2C1_END):
        return True
    return False


def sub_data(n_ratio, m_ratio, w, spec):
    n_1 = int(spec.shape[0]/n_ratio-w)
    n_2 = int(spec.shape[0]/n_ratio+w)
    m_1 = int(spec.shape[1]/m_ratio-w)
    m_2 = int(spec.shape[1]/m_ratio+w)
    return spec[n_1:n_2, m_1:m_2, :]


def remove_data(n_ratio, m_ratio, w, spec):
    n_1 = int(spec.shape[0]/n_ratio-w)
    n_2 = int(spec.shape[0]/n_ratio+w)
    m_1 = int(spec.shape[1]/m_ratio-w)
    m_2 = int(spec.shape[1]/m_ratio+w)
    spec[n_1:n_2, m_1:m_2, :] = float('NaN')
    return spec


def select_wavelength(s_0, s_1):
    if s_1 < 0:
        local_wavelength = GLOBAL_WAVELENGTH[s_0:s_1]
    else:
        local_wavelength = GLOBAL_WAVELENGTH[s_0:]
    return local_wavelength


def select_data(data, s_0=0, s_1=0):
    if s_1 < 0:
        local_data = data[:, :, s_0:s_1]
    else:
        local_data = data[s_0:]
    return local_data


def process_data(spec, filename=None, norm=False, crs=False, smooth=False, smooth_crs=False, exclude_crs=False, marker=float('NaN')):
    proc_filename = None
    if filename is not None:
        name = ''
        if norm:
            name += '_norm'
        if crs:
            name += '_crs'
        if exclude_crs:
            name += '_ex'
        if ~np.isnan(marker):
            name += f'_marker_{marker}'
        proc_filename = filename.split('.npy')[0] + f'{name}_proc.npy'

        # if check_file(proc_filename):
        #     return np.load(proc_filename), proc_filename

    processed_spectra = process_spectra(
        spec, norm, crs, smooth, smooth_crs, exclude_crs, marker)

    # if filename is not None:
    # np.save(proc_filename, processed_spectra)

    return processed_spectra, proc_filename


def process_spectra(spec, norm=False, crs=False, smooth=False, smooth_crs=False, exclude_crs=False, marker=float('NaN')):
    spectra = []
    dim = len(spec.shape) - 1

    if dim == 0:
        spectra = [process_spectrum(
            spec, norm, crs, smooth, smooth_crs, exclude_crs, marker)]
    elif dim == 1:
        for s in spec:
            spectra.append(process_spectrum(
                s, norm, crs, smooth, smooth_crs, exclude_crs, marker))
    else:
        i = 0
        for spectra_2d in spec:
            spectra_1d = []
            for s in spectra_2d:
                spectra_1d.append(process_spectrum(
                    s, norm, crs, smooth, smooth_crs, exclude_crs, marker))
            spectra.append(spectra_1d)

    return np.array(spectra)


def process_spectrum(spectrum, norm=False, crs=False, smooth=False, smooth_crs=False, exclude_crs=False, marker=float('NaN')):
    epsilon = 1e-12
    if check_exclude_data(spectrum):
        return np.full_like(spectrum, float('NaN'))

    if smooth:
        spectrum = try_savgol_filter(spectrum, 15, 10, mode='interp')
    if crs:
        spectrum = try_crs(spectrum)
        if exclude_crs:
            if np.nanmin(spectrum) > 0.99:
                return np.full_like(spectrum, marker)

        if smooth_crs:
            spectrum = try_savgol_filter(spectrum, 6, 5, mode='interp')
    if norm:
        spectrum = normalize_data(spectrum)
        if exclude_crs:
            if np.nanstd(spectrum) > 0.25:
                return np.full_like(spectrum, marker)
    return spectrum


def check_exclude_data(spectrum):
    if np.any(np.isnan(spectrum)):
        return True
    if np.nanmean(spectrum) < 0.01:
        return True
    return False


def try_savgol_filter(spectrum, window_length=7, polyorder=3, mode='interp'):
    try:
        spectrum = savgol_filter(spectrum, window_length, polyorder, mode=mode)
    except:
        pass
        spectrum = np.full_like(spectrum, float('NaN'))
    return spectrum


def try_crs(spectrum, epsilon=1e-12):
    try:
        spectrum = extract_crs((spectrum+epsilon).tolist())
    except:
        pass
        spectrum = np.full_like(spectrum, float('NaN'))
    return spectrum


def normalize_data(data, epsilon=0):
    """Normalizes the data by subtracting the minimum value (ignoring NaNs), 
    adding epsilon, and dividing by the range (maximum - minimum).

    Args:
      data: The input NumPy array.
      epsilon: A small value to add to the numerator to avoid division by zero.

    Returns:
      The normalized NumPy array.
    """
    if np.all(np.isnan(data)):
        return data
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    return (data - data_min + epsilon) / (data_max - data_min)


def display_convex_hull(spectrum):
    wvl = list(GLOBAL_WAVELENGTH[2:-7])

    schq = spectro.SpectrumConvexHullQuotient(spectrum, wvl)
    schq.display('display_name')


def extract_and_display_features(spectrum, baseline):
    wvl = list(GLOBAL_WAVELENGTH[2:-7])
    fea = spectro.FeaturesConvexHullQuotient(spectrum, wvl, baseline=baseline)
    fea.display('display_name', feature='all')
    return fea


def narrow_window(image):
    """
    Get the most narrow minimum and maximum values of an image, while ignoring NaNs.

    :param image: np.array
    :return: xmin, xmax, ymin, ymax
    """
    finite = np.argwhere(np.isfinite(image))
    if not np.any(finite):
        warnings.warn("no finite numbers in array")
        return -1, -1, -1, -1
    xmin = int(np.min(finite[:, 0]))
    xmax = int(np.max(finite[:, 0]))
    ymin = int(np.min(finite[:, 1]))
    ymax = int(np.max(finite[:, 1]))
    return xmin, xmax, ymin, ymax


def get_average_spectra(image):
    """

    :param image: hyperspectral image
    :return: spectra mean, spectra average
    """
    img_av = np.nanmean(image, axis=(0, 1))
    img_std = np.nanstd(image, axis=(0, 1))
    return img_av, img_std


def df_matrix_to_vector(df):
    """
    Convert a 2D array to a 1D dataframe with latitude and longitude values.

    :param df:
    :return: df
    """
    df = df.reset_index()
    df = df.rename(columns={'index': 'latitude'})
    df = df.melt(id_vars=['latitude'],
                 var_name="longitude",
                 value_name="value")
    return df
