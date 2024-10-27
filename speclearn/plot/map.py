import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata

from speclearn.io.data.M3_image import M3_image
from speclearn.tools.cache import *
from speclearn.io.transform.rebin import rebin_from_aoi
from speclearn.tools.constants import M3_PICKLE_DATA_DIR, LONG_DEG, LAT_DEG
from speclearn.tools.map_projections import to_radians, stereographic_projection, loc_2d
from speclearn.io.data.aoi import read_aoi_data, get_loc
from sklearn.cluster import KMeans
from speclearn.io.transform.bin_tools import make_bins


def plot_images(figure_dir, figure_name, reflectance_limits=[], cmap='icefire', long_range=[], lat_range=[],
                image_files=[], individual=False):
    figure_name = f'loc_({long_range[0]}-{long_range[1]
                                          }_{lat_range[0]}-{lat_range[1]})'
    if reflectance_limits:
        refl = '_refl'
    else:
        refl = ''
    if len(lat_range) == 0 or len(long_range) == 0:
        print('Must provide longitude and latitude ranges.')
        return

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    for i, file in enumerate(image_files):
        img = M3_image(file, level=2)
        img.fill()
        longitude = np.array(img.longitude)
        latitude = np.array(img.latitude)

        # select range
        long_select = (longitude >= long_range[0]) & (
            longitude <= long_range[1])
        lat_select = (latitude >= lat_range[0]) & (latitude <= lat_range[1])
        select = long_select & lat_select

        # format data (latitude must be positive for stereographic projection)
        if lat_range[0] < 0:
            latitude = -latitude

        longitude = to_radians(longitude[select])
        latitude = to_radians(latitude[select])
        reflectance = img.reflectance[select]
        if len(reflectance) == 0:
            print('no data in range')

        if len(reflectance_limits) > 0:
            select_refl = (reflectance >= reflectance_limits[0]) & (
                reflectance <= reflectance_limits[1])
            longitude = longitude[select_refl]
            latitude = latitude[select_refl]
            reflectance = reflectance[select_refl]

        r, theta = stereographic_projection(longitude, latitude)

        sns.scatterplot(x=theta, y=r, s=5, cmap=cmap, hue=reflectance, palette='icefire', linewidth=0, edgecolor=None,
                        ax=ax, legend=None, hue_norm=(0, 2))
        if individual:
            plt.savefig(f'{figure_dir}/{img.data_file}{refl}.png', dpi=200)

    plt.savefig(f'{figure_dir}/{figure_name}{refl}.png',
                dpi=200, bbox_inches='tight')
    plt.show()
    return


def get_df_2d(data, loc):
    long, lat = loc_2d(loc['long_center'], loc['lat_center'], data)
    df_2d = pd.DataFrame(
        data=data.T, index=loc['lat_center'], columns=loc['long_center'])
    df_2d = df_2d.sort_index(ascending=False)
    df_2d = df_2d.reindex(sorted(df_2d.columns), axis=1)
    return df_2d


def get_polar_loc(df_2d, area='south'):

    if area == 'south':
        df_2d = df_2d[df_2d.index < -60]
        long_2d = to_radians(df_2d.columns)  # + LAT_DEG
        lat_2d = to_radians(-df_2d.index)
    if area == 'north':
        df_2d = df_2d[df_2d.index > 60]
        long_2d = to_radians(df_2d.columns)
        lat_2d = to_radians(df_2d.index)

    r, theta = stereographic_projection(long_2d, lat_2d)
    return r, theta


def plot_latent(data, aoi, area):
    data[data == 0] = float('NaN')
    no_latent = data.shape[2]
    fig, axs = plt.subplots(ncols=no_latent, subplot_kw={
                            'projection': 'polar'}, figsize=(30, 15))
    for ax, _data in zip(axs, np.moveaxis(data, 2, 0)):
        loc = get_loc(aoi)
        df_2d = get_df_2d(_data, loc)
        r, theta = get_polar_loc(df_2d, area)
        dr = np.fabs((r[0] - r[1]) / 2)
        ax.grid(False)
        ax.set_rlim(np.min(r) - dr, np.max(r) + dr)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.pcolormesh(np.array(theta), np.array(r), df_2d)


def get_indices_from_range(lat_range, long_range):
    lat_edges, lat_centers = make_bins([-90, 90], int(180/0.05))
    long_edges, long_centers = make_bins([-180, 180], int(360/0.05))

    lat_min = lat_range[0]
    lat_max = lat_range[1]
    long_min = long_range[0]
    long_max = long_range[1]

    lat_min_index = np.where(lat_edges == lat_min)[0][0]
    lat_max_index = np.where(lat_edges == lat_max)[0][0]
    long_min_index = np.where(long_edges == long_min)[0][0]
    long_max_index = np.where(long_edges == long_max)[0][0]
    return lat_min_index, lat_max_index, long_min_index, long_max_index


def plot_clusters(k, data, latent, coord, aoi, area):
    fig, axs = plt.subplots(ncols=k, subplot_kw={
                            'projection': 'polar'}, figsize=(30, 15))
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster, cluster_2d, _ = fit_kmeans(kmeans, data, latent, coord)
    loc = get_loc(aoi)
    df_2d = get_df_2d(cluster_2d, loc)
    r, theta = get_polar_loc(df_2d, area)
    dr = np.fabs((r[0] - r[1]) / 2)
    for c, ax in enumerate(axs):
        _data = np.zeros(cluster_2d.shape)
        _data[cluster_2d == c] = 1
        _data[_data == 0] = float('NaN')
        df_2d = get_df_2d(_data, loc)
        ax.grid(False)
        ax.set_rlim(np.min(r) - dr, np.max(r) + dr)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.pcolormesh(np.array(theta), np.array(r), df_2d, cmap='Greys_r')
        ax.set_title(f'cluster = {c}')
    return


def compare_recon(original, recon, aoi, area, min_val=None, max_val=None):
    recon[recon == 0] = float('NaN')
    original[original == 0] = float('NaN')
    _min, _max = plot_min_max(original, min_val, max_val)
    fig, axs = plt.subplots(ncols=2, subplot_kw={
                            'projection': 'polar'}, figsize=(15, 5))

    loc = get_loc(aoi)
    original_df_2d = get_df_2d(original[:, :, 50], loc)
    recon_df_2d = get_df_2d(recon[:, :, 50], loc)

    r, theta = get_polar_loc(original_df_2d, area)
    dr = np.fabs((r[0] - r[1]) / 2)
    axs[0].grid(False)
    axs[0].set_rlim(np.min(r) - dr, np.max(r) + dr)
    axs[0].set_theta_direction(-1)
    axs[0].set_theta_offset(np.pi / 2.0)
    axs[0].pcolormesh(np.array(theta), np.array(
        r), original_df_2d, vmin=_min, vmax=_max)

    axs[1].grid(False)
    axs[1].set_theta_direction(-1)
    axs[1].set_theta_offset(np.pi / 2.0)
    axs[1].set_rlim(np.min(r) - dr, np.max(r) + dr)
    cm = axs[1].pcolormesh(np.array(theta), np.array(r),
                           recon_df_2d, vmin=_min, vmax=_max)
    cax, kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    plt.colorbar(cm, cax=cax)
    return


def plot_map(figure_dir, figure_name='', folder='', aoi=None, cuts=DEFAULT_CUTS,
             refl=False, projection='polar', type='coverage',
             verbose=False, area='', title='', image_files=[], s=2, data=None, palette='icefire', save_fig=False, max_val=None, min_val=None):
    name = get_rebin_path(aoi, cuts, refl)
    aoi_name = f"{name}/{aoi['name']}"

    if type == 'coverage':
        data_file = f"{M3_PICKLE_DATA_DIR}/{folder}/{aoi_name}_count.npy"
    else:
        data_file = f"{M3_PICKLE_DATA_DIR}/{folder}/{aoi_name}_mean.npy"

    if not check_file(data_file):
        rebin_from_aoi(aoi=aoi, cuts=cuts, overwrite=False,
                       verbose=verbose, refl=refl, image_files=image_files)

    if data is None:
        data = np.load(data_file)
        if len(data.shape) > 2:
            data = np.load(data_file)[:, :, s]
    data[data == 0] = float('NaN')
    loc = get_loc(aoi, cuts, refl)

    df_2d = get_df_2d(data, loc)
    r = None
    theta = None
    if projection == 'polar':
        r, theta = get_polar_loc(df_2d, area)
        plot_polar(r, theta, df_2d, palette=palette,
                   max_val=max_val, min_val=min_val)
    else:
        plt.figure(figsize=(20, 6))
        ax = sns.heatmap(df_2d, cmap='magma', xticklabels=20, yticklabels=10)

    if len(figure_name) == 0:
        figure_name = f'{aoi_name}'
        if area == 'south':
            figure_name += '_south'
        if area == 'north':
            figure_name += '_north'
        if projection == 'polar':
            figure_name += '_polar'
        if type == 'coverage':
            figure_name += '_coverage'

    plt.gca().set_title(title)
    check_path(figure_dir)
    if projection == 'polar':
        plt.gca().set_theta_direction(-1)
        plt.gca().set_theta_offset(np.pi / 2.0)
    if save_fig:
        plt.savefig(f'{figure_dir}/{figure_name}.png',
                    dpi=200, bbox_inches='tight')
    print(f'{figure_dir}/{figure_name}.png')
    # plt.gca().set_rlim(0,0.5)

    plt.show()

    return df_2d, data.flatten(), r, theta


def plot_polar(r, theta, data, palette='icefire', max_val=None, min_val=None):
    dr = np.fabs((r[0] - r[1]) / 2)
    plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
    _min, _max = plot_min_max(data, min_val, max_val)
    plt.gca().set_rlim(np.min(r) - dr, np.max(r) + dr)
    plt.gca().set_theta_direction(-1)
    plt.gca().set_theta_offset(np.pi / 2.0)
    plt.pcolormesh(np.array(theta), np.array(r), data,
                   cmap=palette, vmin=_min, vmax=_max)
    plt.colorbar()
    return


def plot_min_max(data, min_val, max_val):
    if max_val is not None:
        _max = max_val
    else:
        _max = np.nanmax(data)
    if min_val is not None:
        _min = min_val
    else:
        _min = np.nanmin(data)
    return _min, _max


def plot_aoi_files(aois, s_1, cuts=DEFAULT_CUTS):
    aoi_img = []
    aoi_min = []
    aoi_max = []
    aoi_loc = []
    names = []
    n_aoi = 0
    for aoi in aois:
        img_file = rebin_from_aoi(aoi, stat='mean', verbose=False, cuts=cuts)
        if img_file is None:
            continue
        data_mean, data_max, data_min, bins = read_aoi_data(img_file)

        aoi_img.append(data_mean[:, :, 0:s_1])
        aoi_min.append(data_min[:, :, 0:s_1])
        aoi_max.append(data_max[:, :, 0:s_1])
        aoi_loc.append(bins)
        names.append(aoi['name'])
        n_aoi = n_aoi + 1

        if n_aoi == 4*5:
            break

    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(
        13, 9), gridspec_kw={'hspace': 0.5})

    for data, ax, name in zip(aoi_img, axs.ravel(), names):
        ax.set_title(name)
        sns.heatmap(data[:, :, 0], ax=ax)
    plt.show()
    return


def predict_kmeans(kmeans, data, latent, coord):
    cluster = kmeans.predict(latent)
    # cluster = cluster.labels_#kmeans.predict(spec_data)

    cluster_2d = np.full_like(data[:, :, 0], float('NaN'))
    for i, (x, y) in enumerate(coord):
        cluster_2d[x, y] = cluster[i]
    return cluster, cluster_2d


def fit_kmeans(kmeans, data, latent, coord):
    cluster = kmeans.fit_predict(latent)
    # cluster = cluster.labels_#kmeans.predict(spec_data)

    cluster_2d = np.full_like(data[:, :, 0], float('NaN'))
    for i, (x, y) in enumerate(coord):
        cluster_2d[x, y] = cluster[i]
    return cluster, cluster_2d, kmeans


def plot_predict_kmeans(data, latent, coord, aoi, area, _kmeans):
    fig, axs = plt.subplots(ncols=1, subplot_kw={
                            'projection': 'polar'}, figsize=(8, 6))
    ax = axs
    # for k, ax in enumerate(axs):
    # kmeans = KMeans(n_clusters=k+3, random_state=0)
    cluster, cluster_2d = predict_kmeans(_kmeans, data, latent, coord)
    loc = get_loc(aoi)
    df_2d = get_df_2d(cluster_2d, loc)
    r, theta = get_polar_loc(df_2d, area)
    dr = np.fabs((r[0] - r[1]) / 2)
    ax.grid(False)
    ax.set_rlim(np.min(r) - dr, np.max(r) + dr)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.pcolormesh(np.array(theta), np.array(r), df_2d, cmap='viridis')
    # ax.set_title(f'nclusters = {k+3}')
    return


def plot_kmeans(data, latent, coord, aoi, area):
    fig, axs = plt.subplots(ncols=5, subplot_kw={
                            'projection': 'polar'}, figsize=(30, 15))

    for k, ax in enumerate(axs):
        kmeans = KMeans(n_clusters=k+3, random_state=0)
        cluster, cluster_2d, kmeans = fit_kmeans(kmeans, data, latent, coord)
        loc = get_loc(aoi)
        df_2d = get_df_2d(cluster_2d, loc)
        r, theta = get_polar_loc(df_2d, area)
        dr = np.fabs((r[0] - r[1]) / 2)
        ax.grid(False)
        ax.set_rlim(np.min(r) - dr, np.max(r) + dr)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.pcolormesh(np.array(theta), np.array(r), df_2d)
        ax.set_title(f'nclusters = {k+3}')
    return
