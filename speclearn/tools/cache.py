import os
import numpy as np

from speclearn.tools.constants import DEFAULT_CUTS, LAT_DEG, LONG_DEG


def check_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return


def check_file(file_path):
    if os.path.exists(file_path):
        return True
    else:
        return False


def get_rebin_path(aoi, cuts=DEFAULT_CUTS, refl=False, n_wave=None, proj=''):
    n_long = int(LONG_DEG/aoi["d_long"])
    n_lat = int(LAT_DEG/aoi["d_lat"])
    bin_info = f"nlong{n_long}_nlat{n_lat}"
    noise = ''
    if refl:
        refl = 'refl'
    else:
        refl = 'spec'
    if len(cuts) > 0:
        cut_name = f'sun_zenith_{int(cuts["sun_zenith"])}_sensor_zenith_{int(
            cuts["sensor_zenith"])}_obs_phase_angle_{cuts["obs_phase_angle"]}/'
    else:
        cut_name = ''
    if len(proj) > 0:
        proj = f'_{proj}'
    if n_wave is not None:
        bin_info += f'_nwave{n_wave}'
    name = f"{bin_info}/{cut_name}{refl}{proj}{noise}"
    return name


def save_files(img_rebin, img_count, img_bins, img_min, img_max, folder, aoi_name, period_name=''):
    """Saves image data to .npy files, handling potential None values."""
    check_path(f"{folder}")

    if img_rebin is not None and img_count is not None:
        print(f'Saving {folder}/{aoi_name}{period_name}_mean.npy')
        np.save(f"{folder}/{aoi_name}{period_name}_mean.npy",
                np.divide(img_rebin, img_count, where=img_count > 0, out=np.full_like(img_rebin, float('NaN'))))

    # Combine the remaining checks into a loop for efficiency
    for data, filename in zip([img_count, img_rebin, img_bins, img_min, img_max],
                              ["_count", "_sum", "_bins", "_min", "_max"]):
        if data is not None:
            print(f'Saving {folder}/{aoi_name}{period_name}{filename}.npy')
            np.save(f"{folder}/{aoi_name}{period_name}{filename}.npy", data)
    return
