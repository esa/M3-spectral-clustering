import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from spectral import *

from speclearn.deep_learning.early_stopping import *
from speclearn.deep_learning.loss_function import SAD, SID
from speclearn.deep_learning.ml_tools import (get_device, load_file_ml, Config)
from speclearn.tools.constants import *
from speclearn.tools.data_tools import *
from speclearn.tools.map_projections import *
import matplotlib
import seaborn as sns
from .models import CAE, CVAE
import torch.nn.functional as F


def get_colorbar_colorblind(k):
    s = sns.color_palette("colorblind", n_colors=10)

    if k == 6:
        c_list = [s[8], s[2], s[0], s[5], s[6], s[-1]]
    if k == 7:
        c_list = [s[8], s[2], s[0], s[5], s[6], s[-1], s[3]]
    if k == 8:
        c_list = [s[8], s[2], s[0], s[5], s[6], s[-1], s[3], s[1]]
    cmap = matplotlib.colors.ListedColormap(c_list, "")
    cmap.set_over('dimgray')
    cmap.set_under('black')

    return c_list, cmap


def get_colorbar(k):
    map = 'viridis'
    n_colors = 200

    palette = sns.color_palette(map, n_colors=n_colors)
    b = [sns.color_palette("Set2")[-1]]

    if k == 8:
        a = [palette[0], palette[30], palette[60], palette[90]]
    elif k == 6:
        a = [palette[0], palette[60]]
    elif k == 5:

        a = [palette[30], palette[100], b[0], palette[160], palette[199]]
    else:
        a = [palette[30], palette[60], palette[90]]

    c = [palette[120], palette[160], palette[199]]

    clist = a[::-1]+b+c[::-1]
    if k == 2:
        clist = [clist[0]]+[clist[-2]]
    if k == 5:
        clist = a
        a = ['#432e6b', '#4580ba', '#b3b3b3', '#7cd250', '#fbeb37']  # fced69

    cmap = matplotlib.colors.ListedColormap(clist, "")
    cmap.set_under('black')

    return clist, cmap


def load_beta_VAE_model(crs=True, norm=True, period='', input_model=None, architecture=2):
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
        "architecture": 2,
        "dropout": 0.1,
    }
    run_name = ''
    if architecture == 2:
        run_name = 'd22kpp8a'
    # if period == 'OP2_C1':
    #     # model_name = '7ka9p7pk'
    # if input_model is not None:
    #     model_name = input_model
    if architecture == 4:
        run_name = 'jbvbrv7u'

    model, criterion, optimizer = make_model(Config(config))
    prev_model = wandb.restore(
        f'{run_name}_model.h5', f'freja-thoresen/M3-autoencoders/{run_name}', replace=True)
    model.load_state_dict(torch.load(prev_model.name))
    return model, run_name


def make_model(config):
    device, use_cuda = get_device()
    bottleneck = False

    if 'B' in config.model:
        bottleneck = True
    if 'VAE' in config.model:
        model = CVAE(no_latent=config.no_latent, input_size=config.input_size, bottleneck=bottleneck,
                     vae=True, dropout=config.dropout, architecture=config.architecture, beta=config.beta)
    else:
        model = CAE(no_latent=config.no_latent, input_size=config.input_size, bottleneck=bottleneck,
                    vae=False, dropout=config.dropout, architecture=config.architecture)

    model.to(device)

    if config.type == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    elif config.type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.loss_function == "SAD":
        criterion = SAD()
    elif config.loss_function == "BCE":
        criterion = nn.BCELoss()
    elif config.loss_function == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = SID()

    return model, criterion, optimizer


def get_batch_data(X_train, batch_size, M3_PICKLE_AOI_DIR, crs=False, norm=False):
    """
    Generator that yields batches of shuffled data with the specified batch size.
    Combines and shuffles data from multiple files. Removes NaN values.

    Args:
        X_train: List of filenames for training data.
        batch_size: The desired batch size.
        M3_PICKLE_AOI_DIR: Directory containing the data files.
        crs: Whether to apply coordinate reference system transformation.
        norm: Whether to normalize the data.

    Yields:
        A batch of shuffled data as a PyTorch tensor.
    """

    file_index = 0
    data_buffer = []
    buffer_size = 0

    # Shuffle the order of files
    random.shuffle(X_train)

    while file_index < len(X_train):
        filename = X_train[file_index]
        _data, data, coord = load_file_ml(
            os.path.join(M3_PICKLE_AOI_DIR, filename),
            as_type='tensor',
            crs=crs,
            norm=norm,
            process=True
        )
        if len(data.shape) == 1:
            file_index += 1
            continue
        # Remove NaN values from data
        data = data[~torch.isnan(data).any(dim=1)]
        if data.shape[1] != 71:
            file_index += 1
            continue

        # Shuffle data within the file
        indices = torch.randperm(data.shape[0])
        data = data[indices]

        data_buffer.append(data)
        buffer_size += data.shape[0]

        if buffer_size >= batch_size:
            if buffer_size == batch_size:
                batch_data = torch.cat(data_buffer, dim=0)
            else:
                batch_data = torch.cat(data_buffer, dim=0)[:batch_size]

            if buffer_size == batch_size:
                data_buffer = []
                buffer_size = 0
            else:
                data_buffer = [batch_data[batch_size:]]
                buffer_size = data_buffer[0].shape[0]

            yield batch_data

        file_index += 1

    if data_buffer:
        yield torch.cat(data_buffer, dim=0)


def train_model(X_train, X_valid, model, criterion, optimizer, config, patience=10, with_wandb=True, crs=True, norm=True):
    random.shuffle(X_train)
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    epochs = config.epochs

    epoch_training_loss = []
    epoch_validation_loss = []

    no_batches = config.no_batches
    batch_data = 0
    best_validation_loss = float('inf')
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        training_loss = 0.
        train_total_batches = 0
        random.shuffle(X_train)
        for batch_data in get_batch_data(X_train, no_batches, M3_PICKLE_AOI_DIR, crs, norm):
            if batch_data.shape[0] != no_batches:
                continue
            optimizer.zero_grad()
            latent, decoded, mu, logVar = model.forward(batch_data)
            loss = model.loss_function(
                decoded, batch_data, mu=mu, logVar=logVar)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            # Increment for each batch
            train_total_batches += batch_data.shape[0]
            # print('train batch loss', loss.item())

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        validation_loss = 0.
        valid_total_batches = 0
        batch_data = 0
        with torch.no_grad():  # No need to calculate gradients during validation
            for batch_data in get_batch_data(X_valid, no_batches, M3_PICKLE_AOI_DIR, crs, norm):
                if batch_data.shape[0] != no_batches:
                    continue
                latent, decoded, mu, logVar = model.forward(batch_data)
                loss = model.loss_function(
                    decoded, batch_data, mu=mu, logVar=logVar)
                validation_loss += loss.item()
                # Increment for each batch
                valid_total_batches += batch_data.shape[0]
                # print('valid batch loss', loss.item())

        # Calculate average losses
        epoch_training_loss.append(training_loss / train_total_batches)
        epoch_validation_loss.append(validation_loss / valid_total_batches)

        if epoch % 1 == 0:
            print(f"epoch : {epoch}/{epochs}, training loss = {
                  epoch_training_loss[-1]}, validation loss = {epoch_validation_loss[-1]}")
        if with_wandb:
            wandb.log({
                'epoch': epoch,
                'validation_loss': epoch_validation_loss[-1],
                'training_loss': epoch_training_loss[-1]
            })

        if epoch > 0 and epoch_validation_loss[-1] < best_validation_loss:
            print(f"Validation loss improved from {best_validation_loss:.10f} to {
                  epoch_validation_loss[-1]:.10f}, saving model...")
            best_validation_loss = epoch_validation_loss[-1]

            torch.save(model.state_dict(), f'{
                       wandb.run.dir}/{wandb.run.id}_model.h5')
            if with_wandb:
                wandb.save(f'{wandb.run.dir}/{wandb.run.id}_model.h5',
                           base_path=f'{wandb.run.dir}')

    return epoch_training_loss, epoch_validation_loss


def check_no_batches(data, no_batches):
    if no_batches > data.shape[0]:
        print(f'Too many batches, data length: {
              data.shape[0]}, no_bathes: {no_batches}')
