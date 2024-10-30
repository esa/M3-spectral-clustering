from speclearn.deep_learning.ml_tools import divide_data
from speclearn.io.data.aoi import get_aoi_list_with_full_path
from speclearn.deep_learning.model_utils import make_model, train_model
import wandb

aoi_list = get_aoi_list_with_full_path(
    '/media/freya/rebin/M3/pickles_v2/nlong7200_nlat3600/train')
X_train, X_valid, X_test = divide_data(aoi_list, 10000)

print('training files', len(X_train))
print('validation files', len(X_valid))
print('test files', len(X_test))

sweep_config = {
    'name': 'beta_CVAE',
    'project': 'M3-autoencoders',
    'method': 'grid',
    'metric': {
        'name': 'training_loss',
        'goal': 'minimize'
    }
}

parameters_dict = {
    'learning_rate': {
        'value': 0.0001
    },
    'epochs': {
        'value': 100
    },
    'no_latent': {
        'value': 6
    },
    'no_batches': {
        'value': 256
    },
    'type': {
        'value': 'AdamW'
    },
    'loss_function': {
        'value': 'MSE'
    },
    'model': {
        'values': ['CAE', 'CVAE']
    },
    'beta': {
        'value': 0.001
    },
    'input_size': {
        'value': 71
    },
    "weight_decay": {
        'value': 0.00001
    },
    "dropout": {
        'value': 0.1
    },
    "architecture": {
        'values': [1, 2, 3]
    }
}

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="M3-autoencoders")


def sweep():
    with wandb.init() as run:
        model, criterion, optimizer = make_model(wandb.config)
        training_loss, validation_loss = train_model(
            X_train, X_valid, model, criterion, optimizer, wandb.config, patience=20, with_wandb=True, crs=True, norm=True)


wandb.agent(sweep_id, function=sweep)
