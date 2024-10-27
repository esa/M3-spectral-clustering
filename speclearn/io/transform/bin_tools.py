import numpy as np

from speclearn.tools.cache import check_file, check_path, get_rebin_path
from speclearn.tools.constants import M3_REBIN_DATA_DIR


def get_limited_range_args(x, x_range):
    limited = np.argwhere((x >= x_range[0]) & (x <= x_range[1])).flatten()
    if len(limited) == 0:
        return -1, -1
    start = limited[0]
    end = limited[-1]
    return start, end


def find_index_in_full(long_centers, lat_centers, img_long, img_lat):
    full_img_long = [np.argwhere(np.around(long_centers, 3) == np.around(img_long[0], 3)).flatten()[0],
                     np.argwhere(np.around(long_centers, 3) == np.around(img_long[1], 3)).flatten()[0]]
    full_img_lat = [np.argwhere(np.around(lat_centers, 3) == np.around(img_lat[0], 3)).flatten()[0],  # +1
                    # +1
                    np.argwhere(np.around(lat_centers, 3) == np.around(img_lat[1], 3)).flatten()[0]]

    return full_img_long, full_img_lat


def limited_img(aoi, binned, bins):
    long = bins['long_center']
    lat = bins['lat_center']
    long = np.around(long, 3)
    lat = np.around(lat, 3)

    long_limit_index = get_limited_range_args(long, aoi['long_range'])
    lat_limit_index = get_limited_range_args(lat, aoi['lat_range'])
    if ((long_limit_index[0] < 0) | (lat_limit_index[0] < 0)):
        return np.array([]), [], []

    binned_limit = binned[long_limit_index[0]:(
        long_limit_index[1]+1), lat_limit_index[0]:(lat_limit_index[1]+1)]
    long_range = [long[long_limit_index[0]], long[long_limit_index[1]]]
    lat_range = [lat[lat_limit_index[0]], lat[lat_limit_index[1]]]

    return binned_limit, long_range, lat_range


def calc_max_bin(img, verbose=False):
    lat = img.latitude.flatten()
    long = img.longitude.flatten()
    lat.sort()
    long.sort()
    d_lat = np.fabs(lat[0:-2] - lat[1:-1]) / 2
    d_long = np.fabs(long[0:-2] - long[1:-1]) / 2
    if verbose:
        print('max lat bin size', np.max(d_lat))
        print('max long bin size', np.max(d_long))
    return np.max(d_long), np.max(d_lat)


def bin_center(bin_edges):
    """
    Convert bin edges to bin centers.

    :param bin_edges: Bin edges
    :return: Bin centers
    """
    return np.array([np.mean([bin_edges[b + 1], bin_edges[b]]) for b in range(0, len(bin_edges[:-1]))])


def make_bins(bin_range, no_bins):
    """
    Make bin edges and bin centers
    """
    bin_edges = np.linspace(bin_range[0], bin_range[1], no_bins + 1)
    bin_centers = bin_center(bin_edges)
    return bin_edges, bin_centers


def get_coord_range(long, lat):
    long_range = [(np.floor(np.nanmin(long))), (np.ceil(np.nanmax(long)))]
    lat_range = [(np.floor(np.nanmin(lat))), (np.ceil(np.nanmax(lat)))]
    return long_range, lat_range


def check_cache(img, aoi, cuts={}, refl=False, proj='', extra=''):
    rebin_path = get_rebin_path(
        aoi=aoi, cuts=cuts, refl=refl, proj=proj)
    # prepare path
    file_path = f'{M3_REBIN_DATA_DIR}/{rebin_path}'
    check_path(file_path)

    file = f'{file_path}/{img.data_file}{extra}.npz'
    exists = check_file(file)
    return file, exists
