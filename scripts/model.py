# In this file i will have
# - class for training parameters
# - class for encoder
# - class for decoder
# - class for classifier
# - class ccvae which:
#       - defines ccvae arquitecture
#       - receive an object of class batch (sup or unsup) 
#       and computes the forward ())
#       - Computes the ELBO
#       - Apendice C
#

from dataclasses import dataclass
import torch.distributions as dist
import torch.nn.functional as F
import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Encoder(nn.Module):
    def __init__(self,
                 z_dim: int,
                 hidden_dim: int = 256,
                 *args,
                 **kwargs):
        super().__init__()
        # Arquitecture from Appendix C.3
        self.z_dim = z_dim
        # Input is 32 x 32 x 3 channnel image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),
            nn.ReLU(True),
            View((-1, hidden_dim*1*1))
        )

        self.locs = nn.Linear(hidden_dim, z_dim)
        self.scales = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.locs(hidden), torch.clamp(F.softplus(self.scales(hidden)), min=1e-3)     


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=256, *args, **kwargs):
        super().__init__()
        # setup the two linear transformations used
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            View((-1, hidden_dim, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        m = self.decoder(z)
        return m

class Diagonal(nn.Module):
    def __init__(self, dim):
        super(Diagonal, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):
        return x * self.weight + self.bias

class Classifier(nn.Module):
    def __init__(self, dim):
        super(Classifier, self).__init__()
        self.dim = dim
        self.diag = Diagonal(self.dim)

    def forward(self, x):
        return self.diag(x)

class CondPrior(nn.Module):
    def __init__(self, dim):
        super(CondPrior, self).__init__()
        self.dim = dim
        self.diag_loc_true = nn.Parameter(torch.zeros(self.dim))
        self.diag_loc_false = nn.Parameter(torch.zeros(self.dim))
        self.diag_scale_true = nn.Parameter(torch.ones(self.dim))
        self.diag_scale_false = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
        scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
        return loc, torch.clamp(F.softplus(scale), min=1e-3)


class CCVAE(nn.Module):
    def __init__(self,
                 z_dim: int,
                 y_prior_params: list,
                 num_classes: int,
                 image_shape: tuple,
                 device,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.z_dim = z_dim
        self.y_prior_params = y_prior_params
        self.num_labeled = num_classes
        self.num_unlabeled = z_dim - num_classes
        self.device = device
        self.image_shape = image_shape
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.classifier = Classifier(self.num_labeled)
        self.conditional_prior = CondPrior(self.num_labeled)

    def supervised_ELBO(self):
        pass

    def unsupervised_ELBO(self, images):
        batch_size = images.shape[0]

        # posterior parameters
        params_phi = self.encoder(images)

        # the approximate posterior
        q_phi_z_x = dist.Normal(*params_phi)
        z = q_phi_z_x.rsample()

        # image reconstruction
        r = self.decoder(z)

        # the first element of the vector z correspond to
        # the caracteristics we want to learn
        zc, zs = z.split([self.num_labeled, self.num_unlabeled], 1)

        # Classification
        p = self.classifier(zc)  # FIXME: better name for variable
        # the label predicted distribution
        q_varphi_y_zc = dist.Bernoulli(logits=p)
        y = q_varphi_y_zc.sample()
        log_q_varphi_y_zc = q_varphi_y_zc.log_prob(y).sum(dim=-1)

        # Conditional Prior
        mu_psi, sigma_psi = self.conditional_prior(y)
        params_psi = (
            torch.cat([mu_psi, torch.zeros(1, self.num_unlabeled).expand(batch_size, -1)], dim=1),
            torch.cat([sigma_psi, torch.ones(1, self.num_unlabeled).expand(batch_size, -1)], dim=1))
        p_psi_z_y = dist.Normal(*params_psi)

        # The prior labeled data
        p = self.y_prior_params.expand(batch_size, -1)  # FIXME: better name for variable
        log_py = dist.Bernoulli(p).log_prob(y).sum(dim=-1)

        # Generative model distribution
        p_theta_x_z = dist.Laplace(r, torch.ones_like(r))
        log_theta_x_z = p_theta_x_z.log_prob(images).sum(dim=(1,2,3))

        # ELBO
        kl = dist.kl.kl_divergence(q_phi_z_x, p_psi_z_y).sum(dim=-1)
        elbo = (log_theta_x_z + log_py - kl - log_q_varphi_y_zc).mean()
        return -elbo