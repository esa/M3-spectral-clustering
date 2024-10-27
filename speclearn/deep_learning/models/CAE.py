
from typing import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


def L_out(L_in, padding, kernel_size, stride, dilation=1):
    return int((L_in + 2*padding - dilation*(kernel_size - 1)-1)/stride + 1)


class CAE(nn.Module):
    def __init__(self, **kwargs):
        # input
        self.no_kernels = 2
        self.no_latent = kwargs["no_latent"]
        self.input_size = kwargs["input_size"]
        self.kernel_size = 3
        self.bottleneck = kwargs['bottleneck']
        self.vae = kwargs['vae']
        if self.vae == True:
            self.latent_multiplier = 2
        else:
            self.latent_multiplier = 1
        self.architecture = kwargs['architecture']

        self.dropout = kwargs['dropout']
        super().__init__()

        self.create_layers()

    def loss_function(self, recon_x, x, **kwargs):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        return recon_loss

    def create_layers(self):
        if self.architecture == 1:
            self.encoder_layers = nn.Sequential(
                nn.Linear(in_features=71, out_features=120),
                nn.BatchNorm1d(120),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),  # Dropout after the first linear layer
                nn.Unflatten(dim=1, unflattened_size=(1, 120)),
                nn.Conv1d(in_channels=1, out_channels=5,
                          kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),  # Dropout after the first Conv1d layer
                nn.Conv1d(in_channels=5, out_channels=1,
                          kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),  # Dropout after the second Conv1d layer
                nn.Flatten(),
                nn.Linear(in_features=120,
                          out_features=self.no_latent*self.latent_multiplier)
            )

            # Decoder
            self.decoder_layers = nn.Sequential(
                nn.Linear(in_features=self.no_latent, out_features=120),
                nn.BatchNorm1d(120),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),  # Dropout after the first linear layer
                nn.Unflatten(dim=1, unflattened_size=(1, 120)),
                nn.ConvTranspose1d(
                    in_channels=1, out_channels=5, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                # Dropout after the first ConvTranspose1d layer
                nn.Dropout(p=0.1),
                nn.ConvTranspose1d(
                    in_channels=5, out_channels=1, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                # Dropout after the second ConvTranspose1d layer
                nn.Dropout(p=0.1),
                nn.Flatten(),
                nn.Linear(in_features=120, out_features=71),
                nn.Sigmoid()
            )
        elif self.architecture == 2:
            # Encoder layers
            self.encoder_layers = nn.Sequential(
                nn.Linear(71, 64),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                # Output both mu and logvar
                nn.Linear(32, self.no_latent * self.latent_multiplier)
            )

            # Decoder layers
            self.decoder_layers = nn.Sequential(
                nn.Linear(self.no_latent, 32),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(32, 64),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(64, 71),
                nn.Sigmoid()  # Output in the range [0, 1]
            )
        elif self.architecture == 3:
            # Encoder layers (with 1D convolutions)
            self.encoder_layers = nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=(1, 71)),
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,
                          stride=2, padding=1),  # Input channels = 1
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Conv1d(in_channels=16, out_channels=32,
                          kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Flatten(),
                # Adjust output size based on conv layers
                nn.Linear(32 * 18, self.no_latent * self.latent_multiplier)
            )

            # Decoder layers (with transposed convolutions)
            self.decoder_layers = nn.Sequential(
                # Adjust input size based on conv layers
                nn.Linear(self.no_latent, 32 * 18),
                # Reshape for transposed convolutions
                nn.Unflatten(1, (32, 18)),
                nn.Dropout(p=0.1),
                nn.ConvTranspose1d(in_channels=32, out_channels=16,
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.ConvTranspose1d(in_channels=16, out_channels=1,
                                   kernel_size=3, stride=2, padding=1, output_padding=0),
                nn.Flatten(),
                nn.Sigmoid()
            )
        elif self.architecture == 4:
            # Encoder layers (with 1D convolutions)
            self.encoder_layers = nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=(1, 71)),
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3,
                          stride=2, padding=1),  # Input channels = 1
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Conv1d(in_channels=16, out_channels=32,
                          kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Flatten(),
                # Adjust output size based on conv layers
                nn.Linear(32 * 18, self.no_latent * self.latent_multiplier)
            )

            # Decoder layers
            self.decoder_layers = nn.Sequential(
                nn.Linear(self.no_latent, 32),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(32, 64),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(64, 71),
                nn.Sigmoid()  # Output in the range [0, 1]
            )
        else:
            raise ValueError("Invalid architecture number")

    def forward(self, x, **kwargs):
        latent = self.encoder_layers(x)
        decoded = self.decoder_layers(latent)
        return latent, decoded, None, None
