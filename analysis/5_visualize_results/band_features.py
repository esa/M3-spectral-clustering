import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from speclearn.deep_learning.model_utils import (get_colorbar,
                                                 load_beta_VAE_model)
from speclearn.deep_learning.predict import (get_full_data,
                                             process_full_map, read_area)
from speclearn.io.data.aoi import get_full_map_aoi
from speclearn.plot.map import *
from speclearn.tools.cache import check_file
from speclearn.tools.data_tools import *
from pysptools.spectro import FeaturesConvexHullQuotient
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
local_wavelength = select_wavelength(s_0=0, s_1=-12)
import datetime

print('Current time: ', datetime.datetime.now())
figure_dir = '/home/freya/Documents/figures/cluster_maps/'
from speclearn.tools.constants import *

sns.set_style('whitegrid')
sns.set_context('notebook')

# %%
k = 5
crs = False
area = 'large'
norm = False
full=False

# %%
if full:
    full_name='_full'
else:
    full_name=''

clist, cmap = get_colorbar(k)

# %%
aois = get_full_map_aoi(d_long=0.05, d_lat=0.05, step_size=10)
model, model_name = load_beta_VAE_model(crs=crs, norm=norm)
print('model:', model_name)

# %%
process_full_map(aois, crs=crs, periods=[], norm=norm)
data_2d_full, coord_full = get_full_data(aois, crs=crs, periods=[], norm=norm)
data_2d, coord, latent, data_2d_s, coord_s, latent_s, data_2d_n, coord_n, latent_n = read_area(aois, model_name, crs, norm, full=full, area=area)

# %%
if check_file(os.path.join(CACHE_CLUSTER, f'{model_name}_{k}_cluster_2d_70_lat{full_name}.npy')):
    cluster_2d = np.load(os.path.join(CACHE_CLUSTER, f'{model_name}_{k}_cluster_2d_70_lat{full_name}.npy'))
    cluster_2d_s = np.load(os.path.join(CACHE_CLUSTER, f'{model_name}_{k}_cluster_2d_50_lat_south{full_name}.npy'))
    cluster_2d_n = np.load(os.path.join(CACHE_CLUSTER, f'{model_name}_{k}_cluster_2d_50_lat_north{full_name}.npy'))

cluster_2d_s = cluster_2d_s[:,0:400]
cluster_2d_n = cluster_2d_n[:,200:]
cluster_2d_full = np.concatenate([cluster_2d_s, cluster_2d, cluster_2d_n], axis=1)

def get_two_largest(numbers):
    max = 0
    second_max = 0
    i_max = 0
    i_second_max = 0
    for i, n in enumerate(numbers):
        if n > max:
            max = n
            i_max = i
    for i, n in enumerate(numbers):
        if n > second_max and n < max:
            second_max = n
            i_second_max = i
    return i_max, i_second_max

# %%
def get_dominant_features(features):
    features_numbers = np.linspace(0, features.get_number_of_kept_features()-1,features.get_number_of_kept_features(),dtype=int)
    depths = [features.get_absorbtion_depth(i) for i in features_numbers]

    first_feature, second_feature = get_two_largest(depths)
    selected_features = [first_feature, second_feature]

    depths = [features.get_absorbtion_depth(i) for i in selected_features]
    absorbtion_wavelengths = [features.get_absorbtion_wavelength(i) for i in selected_features]
    areas = [features.get_area(i) for i in selected_features]
    fwhm = [features.get_full_width_at_half_maximum(i) for i in selected_features]
    return depths, absorbtion_wavelengths, areas, fwhm

# %%
depths_1, absorbtion_wavelengths_1, areas_1, fwhm_1 = [], [], [], []
depths_2, absorbtion_wavelengths_2, areas_2, fwhm_2 = [], [], [], []

for n_long in range(0, data_2d_full.shape[0]):
    for n_lat in range(0, data_2d_full.shape[1]):
        try:
            features = FeaturesConvexHullQuotient((data_2d_full[n_long, n_lat, :]+1e-12).tolist(), local_wavelength.tolist(), normalize=False, baseline=1)
        except:
            continue
            pass
        if features.get_number_of_kept_features() > 0:
            
            depths, absorbtion_wavelengths, areas, fwhm = get_dominant_features(features)
            depths_1.append(1-depths[0])
            depths_2.append(1-depths[1])
            absorbtion_wavelengths_1.append(absorbtion_wavelengths[0])
            absorbtion_wavelengths_2.append(absorbtion_wavelengths[1])
            areas_1.append(areas[0])
            areas_2.append(areas[1])
            fwhm_1.append(fwhm[0])
            fwhm_2.append(fwhm[1])
        else:
            continue

# %%
plt.errorbar(depths_1, depths_2, fmt='o')

# %%
df = pd.DataFrame(columns=['Depth', 'Absorption wavelength', 'Area', 'FWHM', 'Feature'])
df_secondary = pd.DataFrame(columns=['Depth', 'Absorption wavelength', 'Area', 'FWHM', 'Feature'])

# %%
df['Depth'] = depths_1
df['Absorption wavelength'] = absorbtion_wavelengths_1
df['Area'] = areas_1
df['FWHM'] = fwhm_1
df['Feature'] = 'Dominant'

# %%
df_secondary['Depth'] = depths_2
df_secondary['Absorption wavelength'] = absorbtion_wavelengths_2
df_secondary['Area'] = areas_2
df_secondary['FWHM'] = fwhm_2
df_secondary['Feature'] = 'Secondary'

# %%
df = df.append(df_secondary, ignore_index=True)

# %%
df.to_pickle('/home/freya/Documents/Code/cache/band_features.pkl')

# %%
#sns.pairplot(df, hue="Feature")


