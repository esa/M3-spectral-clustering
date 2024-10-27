
from speclearn.deep_learning.models.CAE import CAE
from ..loss_function import VAE_loss
import torch
import torch.nn.functional as F


def reparameterize(mu, logVar):
    std = torch.exp(0.5 * logVar)
    eps = torch.randn_like(std)

    return mu + eps * std


class CVAE(CAE, VAE_loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta = kwargs["beta"]

    def forward(self, x, eval=False):
        latent = self.encoder_layers(x)

        mu = latent[:, 0:self.no_latent]
        logVar = latent[:, self.no_latent:]

        z = reparameterize(mu, logVar)
        decoded = self.decoder_layers(z)
        return latent, decoded, mu, logVar

    def loss_function(self, recon_x, x, mu=None, logVar=None):
        # Reconstruction loss (e.g., MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        kl_loss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        kl_loss = kl_loss  # Normalize by batch size

        # kl_loss = F.kl_div(q.log_prob(mu + 1e-6), p.log_prob(mu + 1e-6), reduction='sum')

        # Total loss with beta weighting
        loss = recon_loss + self.beta * kl_loss
        return loss
