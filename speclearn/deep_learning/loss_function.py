from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import kl as kl
import torch
import torch.nn as nn
from torch.distributions import Normal


class SAD(nn.Module):
    def __init__(self, num_bands: int = 68):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, decoded, target):
        """Spectral Angle Distance Objective
        Implementation based on the mathematical formulation presented in 'https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7061924'

        Params:
            input -> Output of the autoencoder corresponding to subsampled input
                    tensor shape: (batch_size, num_bands)
            target -> Subsampled input Hyperspectral image (batch_size, num_bands)

        Returns:
            angle: SAD between input and target
        """
        try:
            input_norm = torch.sqrt(torch.bmm(
                decoded.view(-1, 1, self.num_bands), decoded.view(-1, self.num_bands, 1)))
            target_norm = torch.sqrt(torch.bmm(
                target.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1)))

            summation = torch.bmm(
                decoded.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
            angle = torch.acos(summation/(input_norm * target_norm))

        except ValueError:
            return 0.0

        return angle


class SID(nn.Module):
    def __init__(self, epsilon: float = 1e5):
        super(SID, self).__init__()
        self.eps = epsilon

    def forward(self, input, target):
        """Spectral Information Divergence Objective
        Note: Implementation seems unstable (epsilon required is too high)
        Implementation based on the mathematical formulation presented in 'https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7061924'

        Params:
            input -> Output of the autoencoder corresponding to subsampled input
                    tensor shape: (batch_size, num_bands)
            target -> Subsampled input Hyperspectral image (batch_size, num_bands)

        Returns:
            sid: SID between input and target
        """
        normalize_inp = (input/torch.sum(input, dim=0)) + self.eps
        normalize_tar = (target/torch.sum(target, dim=0)) + self.eps
        sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) +
                        normalize_tar * torch.log(normalize_tar / normalize_inp))

        return sid


class VAE_loss():
    @staticmethod
    def _data_fidelity_loss(X, X_hat):
        """
        Calculates the data fidelity loss (reconstruction loss).

        Args:
            X: The original input data.
            X_hat: The reconstructed data.

        Returns:
            The data fidelity loss.
        """

        # Using mean squared error loss
        criterion = nn.MSELoss(reduction='none')  # Get per-element loss
        data_fidelity = criterion(X_hat, X)
        return torch.mean(data_fidelity)  # Mean across features

    @staticmethod
    def kl_divergence_loss(mu, std, z):
        """
        Calculates the KL divergence loss.

        Args:
            mu: The mean of the latent distribution.
            std: The standard deviation of the latent distribution.
            z: The sampled latent vector.

        Returns:
            The KL divergence loss.
        """

        p = Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = Normal(mu, std)

        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        kl_loss = log_qz - log_pz
        return torch.mean(kl_loss, dim=1)  # Mean across latent dimensions

    @staticmethod
    def criterion(X, X_hat, mean, std, z):
        """
        Calculates the total loss for the VAE.

        Args:
            X: The original input data.
            X_hat: The reconstructed data.
            mean: The mean of the latent distribution.
            std: The standard deviation of the latent distribution.
            z: The sampled latent vector.

        Returns:
            A dictionary containing the data fidelity loss, KL divergence loss,
            and the total loss.
        """

        data_fidelity_loss = VAE_loss._data_fidelity_loss(X, X_hat)
        kl_divergence_loss = VAE_loss.kl_divergence_loss(mean, std, z)

        loss = data_fidelity_loss + kl_divergence_loss

        losses = {
            "data_fidelity": torch.sum(data_fidelity_loss),
            "kl-divergence": torch.sum(kl_divergence_loss),
            "loss": torch.mean(loss)
        }
        return losses
