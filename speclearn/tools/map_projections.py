import numpy as np
from scipy import optimize
import imageio
from pygifsicle import optimize
import matplotlib as mpl
from speclearn.tools.constants import LONG_DEG, LAT_DEG


def get_colors(no_colors):
    cmap = mpl.cm.get_cmap('viridis', no_colors)
    colors = [mpl.colors.rgb2hex(cmap(c)) for c in range(0, no_colors)]
    return colors


def orthographic_projection(longitude, latitude, longitude_0=0, latitude_0=-90, R=1):
    """
    Orthographic projection.

    :param longitude: Longitude in radians.
    :param latitude: Latitude in radians.
    :param longitude_0: Center of longitude in the image.
    :param latitude_0: Center of latitude in the image.
    :param R: Radius.
    :return: Cartesian x, y.
    """
    x = R * np.cos(latitude) * np.sin(longitude - longitude_0)
    y = R * (np.cos(latitude_0) * np.sin(latitude) - np.sin(latitude_0) * np.cos(latitude) * np.cos(
        longitude - longitude_0))
    return x, y


def eckert_iv_projection(longitude, latitude, longitude_0=0, R=1):
    """
    Orthographic projection.

    :param longitude: Longitude in radians.
    :param latitude: Latitude in radians.
    :param longitude_0: Center of longitude in the image.
    :param R: Radius.
    :return: Cartesian x, y.
    """
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    longitude_0 = np.array(longitude_0)

    def solve_for_theta(theta):
        root = theta + np.sin(theta) * np.cos(theta) + 2 * np.sin(theta) \
            - (2 + np.pi / 2) * np.sin(latitude)
        return root

    theta = optimize.newton(solve_for_theta, np.full_like(latitude, 0.))

    x = 2. / (np.sqrt(4 * np.pi + np.power(np.pi, 2))) * R * \
        (longitude - longitude_0) * (1 + np.cos(theta))
    y = 2 * np.sqrt(np.pi / (4 + np.pi)) * R * np.sin(theta)

    return x, y


def stereographic_projection(longitude, latitude, R=1):
    """
    Stereographic projection.

    :param longitude: Longitude in radians.
    :param latitude: Latitude in radians.
    :param R: Radius.
    :return: Cylindrical r, theta.
    """
    r = 2 * R * np.tan(np.pi / 4 - latitude / 2)
    theta = longitude
    return r, theta


def to_radians(degrees):
    """
    Convert degrees to radians.

    :param degrees: Degrees.
    :return: Radians.
    """
    return degrees * (np.pi / LAT_DEG)


def loc_2d(long, lat, data):
    """
    Make flattenend long and lat into 2d.
    """
    lat = np.tile(lat, data.shape[0])
    long = np.repeat(long, data.shape[1])
    return long, lat


def wrap_degrees(degrees):
    """
    degrees number or array
    """
    if isinstance(degrees, (np.ndarray)):
        degrees[degrees < 0] += LONG_DEG
        degrees[degrees >= LONG_DEG] += -LONG_DEG
    else:
        if degrees < 0:
            degrees += LONG_DEG
        if degrees >= LONG_DEG:
            degrees += -LONG_DEG
    return degrees
