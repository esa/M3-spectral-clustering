import scipy.stats as stats
from sklearn.metrics import davies_bouldin_score
from speclearn.io.data.m3_data import fill_and_select, get_images_in_range, remove_outliers_spectral_line, remove_zeros_and_ones
from speclearn.io.data.m3_data import data_generator
from speclearn.tools.cache import *
from speclearn.tools.constants import DEFAULT_CUTS, FULL_GLOBAL_WAVELENGTH, GLOBAL_WAVELENGTH, M3_PICKLE_DATA_DIR, MAX_REBIN, LOWER_WAVELENGTH_LIMIT
from speclearn.io.transform.bin_tools import *
import random


def rebin_image(img, aoi, stat='mean', wavelength_id=None, sun=False, lat_range=[], long_range=[]):
    """
    Rebin a 2D image.

    :param values: reflectance image
    :param stat: any stat, e.g. 'sum', 'np.nanmean', 'mean'
    :param n_long: bins in the x-direction
    :param n_lat: bins in the y-direction
    :return: binned image, bins dictionary
    """

    long = img.longitude.flatten()
    lat = img.latitude.flatten()
    if sun:
        values = img.obs_phase_angle.flatten()
    elif wavelength_id is not None:
        values = np.squeeze(img.data[:, :, wavelength_id]).flatten()
    else:
        values = img.reflectance.flatten()

    if len(lat_range) == 0 and len(long_range) == 0:
        long_range, lat_range = get_coord_range(long, lat)

    long_bin_edges, long_bin_centers = make_bins(long_range,
                                                 int(np.fabs(long_range[1] - long_range[0]) / aoi['d_long']))
    lat_bin_edges, lat_bin_centers = make_bins(lat_range, int(
        np.fabs(lat_range[1] - lat_range[0]) / aoi['d_lat']))

    binned_image, x_edge, y_edge, binnumber = stats.binned_statistic_2d(long, lat, values, stat,
                                                                        bins=[long_bin_edges, lat_bin_edges])
    binned_image[binned_image == 0] = float('NaN')

    bins = {
        'long_center': long_bin_centers,
        'lat_center': lat_bin_centers,
        'long_edge': long_bin_edges,
        'lat_edge': lat_bin_edges,
    }

    return binned_image, bins


def rebin_image_3d(img, aoi={}, stat=np.nanmean, lat_range=[], long_range=[]):
    """
    Rebin a 2D image.

    :param img: image
    :param stat: any stat, e.g. 'sum', 'np.nanmean', 'mean'
    :return: binned image, bins dictionary
    """

    wl_range = aoi.get('wl_range', [])
    n_long = int(LONG_DEG/aoi["d_long"])
    n_lat = int(LAT_DEG/aoi["d_lat"])

    if len(wl_range) == 0:
        binned_image = []
        for wl in range(LOWER_WAVELENGTH_LIMIT, len(FULL_GLOBAL_WAVELENGTH)):
            binned, bins = rebin_image(
                img, aoi, stat=np.nanmean, wavelength_id=wl, lat_range=lat_range, long_range=long_range)
            binned_image.append(binned)
        binned_image = np.moveaxis(np.array(binned_image), 0,
                                   # move axis, new dim (longitude, latitude, wavelength)
                                   -1)
    else:

        long = np.repeat(img.longitude.flatten(), img.data.shape[2])
        lat = np.repeat(img.latitude.flatten(), img.data.shape[2])
        wavelength = np.tile(
            GLOBAL_WAVELENGTH, img.data.shape[0] * img.data.shape[1])
        values = img.data.flatten()

        long_range, lat_range = get_coord_range(long, lat)
        long_bin_edges, long_bin_centers = make_bins(long_range, n_long)
        lat_bin_edges, lat_bin_centers = make_bins(lat_range, n_lat)
        wl_bin_edges, wl_bin_centers = make_bins(aoi['wl_range'], aoi['n_wl'])

        binned_image, bins, binnumber = stats.binned_statistic_dd(sample=[long, lat, wavelength], values=values,
                                                                  statistic=stat,
                                                                  bins=[long_bin_edges, lat_bin_edges, wl_bin_edges])
        bins = {
            'long_center': long_bin_centers,
            'lat_center': lat_bin_centers,
            'long_edge': long_bin_edges,
            'lat_edge': lat_bin_edges,
            'wl_edge': wl_bin_edges,
            'wl_center': wl_bin_centers,
        }

    binned_image[binned_image == 0] = float('NaN')

    return binned_image, bins


def load_file(file):
    loaded_file = np.load(file, allow_pickle=True)
    binned = loaded_file['data']
    bins = loaded_file['bins'].tolist()
    return binned, bins


def rebin_image_from_aoi(img, aoi, cuts=DEFAULT_CUTS, refl=False, overwrite=False, verbose=False):
    """
    Rebin image from area of interest. The area of interest must be a dictionary with the following contents:
    :param img:
    :param aoi:
    :param cuts:
    :param refl:
    :param overwrite:
    :param verbose:
    :return:
    """
    if verbose:
        print('-- rebin_image_from_aoi')
    file, exists = check_cache(img, aoi, cuts, refl)
    if exists and not overwrite:
        binned, bins = load_file(file)
        if len(binned.shape) > 2:
            if binned.shape[2] == 81:
                os.remove(file)
                overwrite = True
            else:
                return binned_image, bins, file
        else:
            return file

    if verbose:
        print('rebin_image_from_aoi: fill_and_select:', img.img_hdr)
    try:
        img = fill_and_select(img, proj='', refl=refl, cuts=cuts)
    except:
        print('fill_and_select error')
        pass
        return []
    long_range, lat_range = get_coord_range(img.longitude, img.latitude)
    long_bins = int(np.fabs(long_range[1] - long_range[0]) / aoi['d_long'])
    lat_bins = int(np.fabs(lat_range[1] - lat_range[0]) / aoi['d_lat'])
    long_low = long_range[0]
    lat_low = lat_range[0]

    f = 0
    _file = file.split('.')
    files = []
    print(f'rebin_image_from_aoi: long_bins {long_bins}, lat_bins {lat_bins}')

    if (long_bins > MAX_REBIN) or (lat_bins > MAX_REBIN):

        for lat_bin in range(0, lat_bins, MAX_REBIN):
            for long_bin in range(0, long_bins, MAX_REBIN):
                if f == 0:
                    _file_name = f'{_file[0]}.{_file[1]}'
                else:
                    _file_name = f'{_file[0]}_{f}.{_file[1]}'
                print(f'rebin_image_from_aoi: file {_file_name}')

                long_low = long_range[0] + long_bin*aoi['d_long']
                lat_low = lat_range[0] + lat_bin*aoi['d_lat']
                long_high = long_low + MAX_REBIN*aoi['d_long']
                lat_high = lat_low + MAX_REBIN*aoi['d_lat']

                if check_file(_file_name):
                    if not overwrite:
                        files.append(_file_name)
                        f = f + 1
                        print(f'rebin_image_from_aoi: {_file_name} exists')
                        continue
                    else:
                        os.remove(_file_name)

                if long_high < long_range[1]:
                    _long_range = [long_low, long_high]
                else:
                    _long_range = [long_low, long_range[1]]
                if lat_high < lat_range[1]:
                    _lat_range = [lat_low, lat_high]
                else:
                    _lat_range = [lat_low, lat_range[1]]

                if refl:
                    _binned_image, _bins = rebin_image(img, aoi, stat=np.nanmean,
                                                       lat_range=_lat_range,
                                                       long_range=_long_range)
                else:
                    _binned_image, _bins = rebin_image_3d(img, aoi, stat=np.nanmean,
                                                          lat_range=_lat_range,
                                                          long_range=_long_range)

                if verbose:
                    print('rebin_image_from_aoi', _file_name)

                np.savez_compressed(_file_name, data=_binned_image, bins=_bins)
                files.append(_file_name)

                f = f + 1
                _binned_image = None

        return files
    if refl:
        binned_image, bins = rebin_image(img, aoi, stat=np.nanmean,
                                         lat_range=lat_range,
                                         long_range=long_range)
    else:
        binned_image, bins = rebin_image_3d(img, aoi, stat=np.nanmean,
                                            lat_range=lat_range,
                                            long_range=long_range)

    np.savez_compressed(file, data=binned_image, bins=bins)

    return file


def rebin_image_from_generator(data_gen, aoi, cuts=DEFAULT_CUTS, refl=False, overwrite=False, proj='', verbose=False):
    """
    Rebin image from a data generator. If n_wave is not set, then the dimensions in the z-direction will remain the same.
    rebin_image_from_aoi
    """
    if verbose:
        print('-- rebin_image_from_generator')
    files = []

    for img in data_gen:
        print('rebin_image_from_generator: ___________ IMAGE ___________')
        f = 1
        file, exists = check_cache(
            img, aoi=aoi, cuts=cuts, refl=refl, proj=proj)
        files.append(file)
        if exists:
            found_more = True

            while found_more:
                _file, _exists = check_cache(
                    img, aoi, cuts, refl, proj=proj, extra=f'_{f}')
                if _exists:
                    files.append(_file)
                    f = f+1
                else:
                    found_more = False
        print('rebin_image_from_generator: hdr', img.img_hdr)
        print('rebin_image_from_generator: file', file)

        # continue if file exists
        if exists and not overwrite:
            print('found files')
            continue

        file = rebin_image_from_aoi(
            img, aoi, cuts, refl, overwrite, verbose=verbose)  # binned_image, bins,
        if img.data is not None:
            img.data = None  # release from memory
        if isinstance(file, list):  # (len(binned_image) == 0) and
            files = files + file

    return files


def rebin_from_aoi(aoi, cuts=DEFAULT_CUTS, overwrite=False, refl=False, periods=[], stat='mean', proj='', image_files=[], verbose=False, combine_files=True,  remove_spectral_outliers=True, remove_zeros_from_spectra=True, suffix=''):
    """
    rebin_image_from_generator
    combine_data_files
    save_files
    """
    if verbose:
        print('-- rebin_from_aoi')
    rebin_path = get_rebin_path(aoi, cuts, refl)
    aoi_name = f"{aoi['name']}"
    if verbose:
        print('rebin_from_aoi: rebin_path', rebin_path)
    if len(periods) > 0:
        period_name = '_'+'_'.join(periods)
    else:
        period_name = ''
    pickle = f"{
        M3_PICKLE_DATA_DIR}/{rebin_path}/{aoi_name}{period_name}{suffix}_{stat}.npy"
    print('Preparing pickle:', pickle)
    check_path(f"{M3_PICKLE_DATA_DIR}/{rebin_path}")

    if verbose:
        print(f"rebin_from_aoi: file {pickle}")
    if check_file(pickle) and not overwrite:
        return pickle

    # look for files in range
    if len(image_files) == 0:
        image_files = get_images_in_range(
            aoi['long_range'], aoi['lat_range'], periods=periods, verbose=verbose)
    else:
        if verbose:
            print('rebin_from_aoi: image_files provided.')
    random.shuffle(image_files)
    # make data generator
    data_gen = data_generator(image_files)

    # rebin
    files = rebin_image_from_generator(data_gen, aoi, cuts=cuts, overwrite=overwrite,
                                       refl=refl, proj=proj, verbose=verbose)
    if combine_files:
        pickle = combine_data_files(aoi, files, refl=refl, folder=f"{M3_PICKLE_DATA_DIR}/{rebin_path}", aoi_name=aoi_name,
                                    period_name=period_name, remove_spectral_outliers=remove_spectral_outliers, remove_zeros_from_spectra=remove_zeros_from_spectra, suffix=suffix)
        return pickle
    else:
        return None


def combine_data_files(aoi, files, refl=False, remove_zeros_from_spectra=True, remove_spectral_outliers=True, folder='', aoi_name='', period_name='', suffix=''):
    print('-- combine_data_files')
    # init full image of aoi
    long_bin = int(
        (aoi['long_range'][1] - aoi['long_range'][0]) / aoi['d_long'])
    lat_bin = int((aoi['lat_range'][1] - aoi['lat_range'][0]) / aoi['d_lat'])
    long_edges, long_centers = make_bins(aoi['long_range'], long_bin)
    lat_edges, lat_centers = make_bins(aoi['lat_range'], lat_bin)

    # binned, bins = load_file(files[0])

    if refl:
        img_count = np.zeros((long_bin, lat_bin))
    else:
        img_count = np.zeros((long_bin, lat_bin, len(GLOBAL_WAVELENGTH)))

    stats = ['mean']  # , 'min', 'max']
    file_counter = 0
    for stat in stats:
        img_stat = np.full_like(img_count, float('NaN'))

        for file in files:
            print(f'File number {file_counter} out of {len(files)} no. files')
            print(f'Starting to combine {file}')
            file_counter += 1
            if not os.path.exists(file):
                print(f'file {file} does not exist.')
                continue
            try:
                binned, bins = load_file(file)
            except:
                print('file broken?')
                # os.remove(file)
                continue

            print('-- limited_img')
            # limit image to aoi limits
            binned, long_range, lat_range = limited_img(aoi, binned, bins)
            if len(binned) == 0:
                continue

            print('-- remove zeros')
            if remove_zeros_from_spectra:
                binned, _ = remove_zeros_and_ones(binned)
            print('-- remove outliers')
            if remove_spectral_outliers:
                binned, _ = remove_outliers_spectral_line(binned)

            print('-- find_index_in_full')

            # get index of image in aoi
            img_long_arg, img_lat_arg = find_index_in_full(
                long_centers, lat_centers, long_range, lat_range)
            print('-- mean')

            # add image to aoi
            if stat == 'mean':
                img_stat[img_long_arg[0]:(img_long_arg[1]+1), img_lat_arg[0]:(img_lat_arg[1]+1), ...] = np.nansum(
                    [binned, img_stat[img_long_arg[0]:(img_long_arg[1]+1), img_lat_arg[0]:(img_lat_arg[1]+1), ...]], axis=0)

                binned_count = np.full_like(binned, 0.)
                binned_count[binned > 0] += 1
                img_count[img_long_arg[0]:(img_long_arg[1]+1), img_lat_arg[0]:(img_lat_arg[1]+1), ...] = np.nansum(
                    [binned_count, img_count[img_long_arg[0]:(img_long_arg[1]+1), img_lat_arg[0]:(img_lat_arg[1]+1), ...]], axis=0)
            elif stat == 'min':
                img_stat[img_long_arg[0]:(img_long_arg[1]+1), img_lat_arg[0]:(img_lat_arg[1]+1), ...] = np.nanmin(
                    [binned, img_stat[img_long_arg[0]:(img_long_arg[1]+1), img_lat_arg[0]:(img_lat_arg[1]+1), ...]], axis=0)
            else:
                img_stat[img_long_arg[0]:(img_long_arg[1]+1), img_lat_arg[0]:(img_lat_arg[1]+1), ...] = np.nanmax(
                    [binned, img_stat[img_long_arg[0]:(img_long_arg[1]+1), img_lat_arg[0]:(img_lat_arg[1]+1), ...]], axis=0)
            print(f'Finished combining {file}')

        if stat == 'mean':
            print(f'Saving {folder}/{aoi_name}{period_name}{suffix}_mean.npy')
            np.save(f"{folder}/{aoi_name}{period_name}{suffix}_mean.npy",
                    np.divide(img_stat, img_count, where=img_count > 0, out=np.full_like(img_stat, float('NaN'))))
            print(f'Saving {folder}/{aoi_name}{period_name}{suffix}_count.npy')
            np.save(
                f"{folder}/{aoi_name}{period_name}{suffix}_count.npy", img_count)
            # print(f'Saving {folder}/{aoi_name}{period_name}_sum.npy')
            # np.save(f"{folder}/{aoi_name}{period_name}_sum.npy", img_stat)
        if stat == 'max':
            print(f'Saving {folder}/{aoi_name}{period_name}_max.npy')
            np.save(f"{folder}/{aoi_name}{period_name}_max.npy", img_stat)
        if stat == 'min':
            print(f'Saving {folder}/{aoi_name}{period_name}_min.npy')
            np.save(f"{folder}/{aoi_name}{period_name}_min.npy", img_stat)
    bins = {
        'long_center': long_centers,
        'lat_center': lat_centers,
        'long_edge': long_edges,
        'lat_edge': lat_edges,
    }

    return f"{folder}/{aoi_name}{period_name}_mean.npy"


def get_rebin_files_from_aoi(aoi, refl=True):
    image_files = get_images_in_range(
        aoi['long_range'], aoi['lat_range'], periods=[], verbose=False)
    data_gen = data_generator(image_files)
    files = rebin_image_from_generator(data_gen, aoi, refl=refl)
    return files
