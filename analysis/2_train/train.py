import wandb

from speclearn.deep_learning.ml_tools import Config, divide_data, load_file_ml, get_aoi_dataset
from speclearn.deep_learning.model_utils import make_model, train_model
from speclearn.io.data.aoi import get_aoi_list_with_full_path
import torch

from speclearn.io.transform.rebin import rebin_from_aoi


X_train, X_valid, X_test = get_aoi_dataset(plot=False)
print('training files', len(X_train))
print('validation files', len(X_valid))
print('test files', len(X_test))

# aoi_list = get_aoi_list_with_full_path('/media/freya/rebin/M3/pickles_v2/nlong7200_nlat3600/train')
# X_train, X_valid, X_test = divide_data(aoi_list, 64800)

with_wandb = True

config = {
    "model": 'CVAE',
    "learning_rate": 0.00001,
    "epochs": 3000,
    "no_batches": 256,
    "no_latent": 5,
    "type": 'AdamW',
    "loss_function": 'MSE',
    "input_size": 71,
    "beta": 0.001,
    "patience": 50,
    "weight_decay": 0.000001,
    'architecture': 4,
    "activation": 'sigmoid',
    "dropout": 0.1,
}

if with_wandb:
    wandb.init(project='M3-autoencoders',
               entity="freja-thoresen", config=config)
    wandb.config = config
    run_config = wandb.config
else:
    run_config = config

model, criterion, optimizer = make_model(Config(config))

# print(wandb.run.dir)
# run_name='a8h7e21r'

# prev_model = wandb.restore(
#     f'{run_name}_model.h5', f'freja-thoresen/M3-autoencoders/{run_name}', replace=True)

# model.load_state_dict(torch.load(prev_model.name,map_location=torch.device('cpu')))

if len(X_train) > 0 and len(X_valid) > 0:
    training_loss, validation_loss = train_model(X_train, X_valid, model, criterion, optimizer, Config(
        run_config), patience=config['patience'], with_wandb=with_wandb, crs=False, norm=False)
    if with_wandb:
        wandb.finish()
