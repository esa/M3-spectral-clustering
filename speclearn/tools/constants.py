import datetime
import numpy as np
import os
import glob

M3_DATA_DIR = '/media/freya/rebin/M3'
M3_PICKLE_DATA_DIR = '/media/freya/rebin/M3/pickles'
M3_PICKLE_AOI_DIR = '/media/freya/rebin/M3/pickles/aoi'
M3_REBIN_DATA_DIR = '/media/freya/rebin/M3/rebin'
M3_PICKLE_DIR = '/media/freya/rebin/M3/pickles'

LAT_DEG = 180
LONG_DEG = 360

FIGURE_DIR = '/home/freya/Documents/Code/spectral-deep-learning/analysis/figures'

FULL_GLOBAL_WAVELENGTH = np.array(
    [460.9900, 500.9200, 540.8400, 580.7600, 620.6900, 660.6100, 700.5400, 730.4800, 750.4400, 770.4000, 790.3700, 810.3300, 830.2900, 850.2500, 870.2100, 890.1700, 910.1400, 930.1000, 950.0600, 970.0200,
     989.9800, 1009.950, 1029.910, 1049.870, 1069.830, 1089.790, 1109.760, 1129.720, 1149.680, 1169.640, 1189.600, 1209.570, 1229.530, 1249.490, 1269.450, 1289.410, 1309.380, 1329.340, 1349.300, 1369.260,
     1389.220, 1409.190, 1429.150, 1449.110, 1469.070, 1489.030, 1508.990, 1528.960, 1548.920, 1578.860, 1618.790, 1658.710, 1698.630, 1738.560, 1778.480, 1818.400, 1858.330, 1898.250, 1938.180, 1978.100,
     2018.020, 2057.950, 2097.870, 2137.800, 2177.720, 2217.640, 2257.570, 2297.490, 2337.420, 2377.340, 2417.260, 2457.190, 2497.110, 2537.030, 2576.960, 2616.880, 2656.810, 2696.730, 2736.650, 2776.580,
     2816.500, 2856.430, 2896.350, 2936.270, 2976.200])

LOWER_WAVELENGTH_LIMIT = 2
GLOBAL_WAVELENGTH = FULL_GLOBAL_WAVELENGTH[LOWER_WAVELENGTH_LIMIT:]

d_bin = (GLOBAL_WAVELENGTH[1:-1] - GLOBAL_WAVELENGTH[0:-2]) / 2
d_bin = np.append(np.append([d_bin[0]], d_bin), [d_bin[-1]])
edges = np.append(GLOBAL_WAVELENGTH - d_bin,
                  [GLOBAL_WAVELENGTH[-1] + d_bin[-1]])
GLOBAL_WAVELENGTH_EDGES = edges
MAX_REBIN = 5000

# DEFAULT_CUTS = {
#     'sun_azimuth': LONG_DEG,
#     'sun_zenith': 90,
#     'sensor_azimuth': LONG_DEG,
#     'sensor_zenith': 25,
#     'obs_phase_angle': 180,
# }

DEFAULT_CUTS = {}

OP1A_START = datetime.datetime(2008, 11, 18)
OP1A_END = datetime.datetime(2009, 1, 24)

OP1B_START = datetime.datetime(2009, 1, 9)
OP1B_END = datetime.datetime(2009, 2, 14)

OP2A_START = datetime.datetime(2009, 4, 15)
OP2A_END = datetime.datetime(2009, 4, 27)

OP2B_START = datetime.datetime(2009, 5, 13)
OP2B_END = datetime.datetime(2009, 5, 16)

OP2C_START = datetime.datetime(2009, 5, 20)
OP2C_END = datetime.datetime(2009, 8, 16)

OP2C1_START = datetime.datetime(2009, 6, 23)
OP2C1_END = datetime.datetime(2009, 7, 22)


def get_L2_files():
    hdr_files = glob.glob(
        os.path.join(
            '/media/freya/*/data/m3/CH1M3_0004/DATA/*/*/L2/',
            'M3G*RFL.HDR'
        )
    )
    return hdr_files


L2_FILES = get_L2_files()


def normalize_data(data, epsilon=0):
    return (data - np.nanmin(data)+epsilon) / (np.nanmax(data) - np.nanmin(data))


CACHE_TIFF = '/home/freya/Documents/Code/cache/tiff'
CACHE_CLUSTER = '/home/freya/Documents/Code/cache/cluster_2d'
CACHE_CLUSTER_SCORES = '/home/freya/Documents/Code/cache/cluster_scores'
CACHE_KMEANS = '/home/freya/Documents/Code/cache/kmeans'
CACHE_GEOJSON = '/home/freya/Documents/Code/cache/geojson'
CACHE_PREDICT = '/home/freya/Documents/Code/cache/predict_v2'
