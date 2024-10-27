import numpy as np
from speclearn.tools.constants import M3_PICKLE_DATA_DIR, DEFAULT_CUTS
from speclearn.tools.cache import get_rebin_path
import random
import os


def read_aoi_data(filename):
    prefix = '_'.join(filename.split('.')[0].split('_')[0:-1])

    mean_suffix = '_mean.npy'
    max_suffix = '_max.npy'
    min_suffix = '_min.npy'
    bins_suffix = '_bins.npy'
    data_mean = np.load(prefix + mean_suffix)
    data_min = np.load(prefix + min_suffix)
    data_max = np.load(prefix + max_suffix)
    bins = np.load(prefix + bins_suffix, allow_pickle=True).tolist()

    return data_mean, data_max, data_min, bins


def get_loc(aoi, cuts=DEFAULT_CUTS, refl=False):
    name = get_rebin_path(aoi, cuts, refl)
    aoi_name = f"{name}/{aoi['name']}"
    loc_file = f"{M3_PICKLE_DATA_DIR}/{aoi_name}_bins.npy"
    loc = np.load(loc_file, allow_pickle=True).tolist()
    return loc


def get_full_map_aoi(d_long=0.05, d_lat=0.05, step_size=10):
    limit = -100
    aois = []
    while limit < 90 - step_size:
        limit += step_size
        aoi_mid = {
            'name': f'loc_-180_180_{int(limit)}_{int(limit+step_size)}',
            'lat_range': [limit, limit+step_size],
            'long_range': [-180.0, 180.0],
            'd_lat': d_lat,
            'd_long': d_long
        }
        aois.append(aoi_mid)
    return aois


def get_full_map_aoi_longitude(d_long=0.05, d_lat=0.05, step_size=10, periods=[]):
    """
    Generates a list of AOIs (Areas of Interest) spanning the full longitude range (-180 to 180 degrees)
    with a specified step size in longitude.

    Args:
      d_long: Longitude resolution (default: 0.05 degrees).
      d_lat: Latitude resolution (default: 0.05 degrees).
      step_size: Step size for longitude in degrees (default: 10 degrees).

    Returns:
      A list of dictionaries, where each dictionary represents an AOI with keys:
        'name': Name of the AOI (e.g., 'loc_-180_-170_-90_90').
        'lat_range': Latitude range as a list [-90, 90].
        'long_range': Longitude range as a list [long_start, long_end].
        'd_lat': Latitude resolution.
        'd_long': Longitude resolution.
    """
    limit = -180
    aois = []
    while limit <= 180 - step_size:

        aoi_mid = {
            'name': f'loc_{int(limit)}_{int(limit+step_size)}_-90_90',
            'lat_range': [-90.0, 90.0],
            'long_range': [limit, limit+step_size],
            'd_lat': d_lat,
            'd_long': d_long
        }
        aois.append(aoi_mid)
        limit += step_size
    return aois


def get_aoi_list_with_full_path(dir_path):
    """
    Returns a list of AOIs with their full paths.

    Args:
      dir_path: The directory containing the AOI files.

    Returns:
      A list of strings, where each string is the full path to an AOI file.
    """
    aoi_list = []
    for filename in os.listdir(dir_path):
        aoi_list.append(os.path.join(dir_path, filename))
    return aoi_list


def make_aoi_at_random(n):
    """
    Creates n random non-overlapping areas of interest (AOIs) 
    with specified latitude and longitude ranges.

    Args:
      n: The number of AOIs to create.

    Returns:
      A list of n AOI dictionaries.
    """

    aoi_list = []
    existing_ranges = []  # Keep track of existing ranges

    for _ in range(n):
        while True:
            lat_start = random.randint(-90, 87)
            lat_end = lat_start + 3
            long_start = random.randint(-180, 177)
            long_end = long_start + 3

            # Check for overlap with existing ranges
            overlap = False
            for existing_lat_range, existing_long_range in existing_ranges:
                if (existing_lat_range[0] <= lat_start <= existing_lat_range[1] or
                    existing_lat_range[0] <= lat_end <= existing_lat_range[1]) and \
                   (existing_long_range[0] <= long_start <= existing_long_range[1] or
                        existing_long_range[0] <= long_end <= existing_long_range[1]):
                    overlap = True
                    break

            if not overlap:
                existing_ranges.append(
                    ([lat_start, lat_end], [long_start, long_end]))
                break

        aoi = {
            'name': f'loc_{long_start}_{long_end}_{lat_start}_{lat_end}',
            'lat_range': [lat_start, lat_end],
            'long_range': [long_start, long_end],
            'd_long': 0.05,
            'd_lat': 0.05,
        }
        aoi_list.append(aoi)

    return aoi_list


def save_npy_subfiles(full_data_path, output_dir, subfile_shape=(20, 20, 81)):
    """
    Saves sub-files of a large NumPy array as individual .npy files.

    Args:
      full_data_path: Path to the .npy file containing the full data array.
      output_dir: Directory to save the sub-files.
      subfile_shape: Shape of each sub-file.
    """

    full_data = np.load(full_data_path)
    full_shape = full_data.shape

    if len(full_shape) != 3 or full_shape[2] != subfile_shape[2]:
        raise ValueError(
            "Incompatible shapes between full data and sub-files.")

    n_rows = full_shape[0] // subfile_shape[0]
    n_cols = full_shape[1] // subfile_shape[1]

    for i in range(n_rows):
        for j in range(n_cols):
            row_start = i * subfile_shape[0]
            row_end = row_start + subfile_shape[0]
            col_start = j * subfile_shape[1]
            col_end = col_start + subfile_shape[1]

            sub_data = full_data[row_start:row_end, col_start:col_end, :]
            filename = f"subfile_{i}_{j}.npy"
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, sub_data)
