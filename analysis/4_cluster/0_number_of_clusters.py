
import torch
import wandb

from speclearn.deep_learning.ml_tools import Config

import datetime
import warnings
import numpy as np
from speclearn.io.data.aoi import get_full_map_aoi
from sklearn.cluster import KMeans
from speclearn.deep_learning.cluster import flag_latent
from speclearn.plot.map import *
from speclearn.tools.data_tools import *
from speclearn.tools.constants import *
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
local_wavelength = select_wavelength(s_0=0, s_1=-12)
from speclearn.deep_learning.predict import get_data, get_full_data, get_names, predict_full_map
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from speclearn.deep_learning.model_utils import load_beta_VAE_model, make_model

print('Current time: ', datetime.datetime.now())
from speclearn.io.data.aoi import get_aoi_list_with_full_path, get_full_map_aoi_longitude

aois = get_full_map_aoi_longitude(step_size=20)


crs = False
norm = False
no_latent = 6
config = {
    "model": "CVAE",
    "learning_rate": 0.00001,
    "epochs": 100,
    "no_batches": 256,
    "no_latent": 6,
    "type": 'Adam',
    "loss_function": 'MSE',
    "input_size": 71,
    "beta": 1,
    "patience": 5,
    "architecture": 2,
    "activation": 'sigmoid',
    "dropout": 0.5,
    "architecture": 4
}
run_name='tlf5peex'
model_name=run_name

model, criterion, optimizer = make_model(Config(config))

prev_model = wandb.restore(
    f'{run_name}_model.h5', f'freja-thoresen/M3-autoencoders/{run_name}', replace=True) 

model.load_state_dict(torch.load(prev_model.name,map_location=torch.device('cpu')))

print('model:', model_name)
for full in [True]:
    #model, model_name = load_beta_VAE_model(crs=crs, norm=norm, architecture=2)
    predict_full_map(aois, model, model_name, crs=crs, norm=norm)
    #data_2d_s, coord_s, latent_s, recon = get_data(aois, model_name, crs=crs, norm=norm)
    aoi_list = get_full_map_aoi_longitude(step_size=20)

    data_2d, coord = get_full_data(aoi_list, crs, norm, periods=[])

    recon = None
    coord_s = None

    if not full:
        # flag the data 
        data_2d_s[:,400:-400,:] = 10
        print('Flagging')
        latent_s = flag_latent(latent_s, data_2d_s)
        
    data_2d_s = None
    scores = {}
    scores['silhuette'] = np.zeros((8))
    scores['davies'] = np.zeros((8))
    scores['calinski'] = np.zeros((8))
    scores['wss'] = np.zeros((80))
    wss = []
    for k in range(2,10):
        print('Clustering for k = ',k)
        kmeans = KMeans(n_clusters=k).fit(latent_s[:,:])
        labels = kmeans.labels_

        print('Calculated scores for k = ',k)
        scores['silhuette'][k-2] = silhouette_score(latent_s[::1000,:], labels[::1000], metric='euclidean')
        scores['davies'][k-2] = davies_bouldin_score(latent_s[::1000,:], labels[::1000])
        scores['calinski'][k-2] = calinski_harabasz_score(latent_s[::1000,:], labels[::1000])

        wss_iter = kmeans.inertia_
        wss.append(wss_iter)
        scores['wss'][k-2] = wss_iter

    print(scores)
    if full:
        full_name = '_full'
    else:
        full_name = ''
    np.save(os.path.join(CACHE_CLUSTER_SCORES, f'cluster_scores_{model_name}{full_name}.npy'), scores)


for aoi in aois:
    print(aoi)
    lat_range = aoi['lat_range']
    latitude_range = f'{int(lat_range[0])}_{int(lat_range[1])}'
    longitude_range = f'{int(aoi["long_range"][0])}_{int(aoi["long_range"][1])}'
            
    crs_name, norm_name, period_name = get_names(
        crs=crs, norm=norm, periods=[])
    
    data_input_file = f'{CACHE_PREDICT}/data_{longitude_range}_{latitude_range}{period_name}{norm_name}{crs_name}.npy'
    recon_file = f'{CACHE_PREDICT}/{model_name}_latent_{longitude_range}_{latitude_range}{period_name}.npy'
    recon_file_crs = f'{CACHE_PREDICT}/{model_name}_recon_{longitude_range}_{latitude_range}{period_name}_crs.npy'

    if check_file(recon_file) and crs:
        if not check_file(recon_file_crs):
            recon_crs = process_data(np.load(recon_file), filename=recon_file,
                           exclude_crs=True, marker=float('NaN'), crs=crs, norm=norm)
            recon_crs = process_data(np.load(recon_file), filename=recon_file,
                           exclude_crs=False, marker=float('NaN'), crs=crs, norm=norm)
            
