import numpy as np
import torch.distributions as dist
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os

from scripts.model_sub_architectures import Classifier, CondPrior, Decoder, Encoder


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

    def accuracy(
            self,
            test_data_loader: DataLoader,
            *args,
            **kwargs):
        acc = 0
        for (image, label) in test_data_loader:
            batch_acc = self.classification_accuracy(
                image.to(device=self.device),
                label.to(device=self.device)
            )
            acc += batch_acc
        return acc

    def classification_accuracy(self,
                                x: torch.tensor,
                                y: torch.tensor):
        # Classificate x using Monte Carlo with only one sample:
        params_phi = self.encoder(x)
        z = dist.Normal(*params_phi).rsample()
        zc, _ = z.split([self.num_labeled, self.num_unlabeled], -1)
        logits = self.classifier(zc.view(-1, self.num_labeled))
        p = torch.round(torch.sigmoid(logits))
        return p.eq(y).float().mean()

    def classifier_loss(self,
                        x: torch.tensor,
                        y: torch.tensor,
                        k: int = 100):
        # Obtain k samples of  the latent space
        params_phi = self.encoder(x)
        z = dist.Normal(*params_phi).rsample(torch.tensor([k]))
        zc, _ = z.split([self.num_labeled, self.num_unlabeled], -1)

        # The label predicted distribution
        logits_c = self.classifier(zc.view(-1, self.num_labeled))
        q_varphi_y_zc = dist.Bernoulli(logits=logits_c)

        # Classifier loss
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_labeled)
        log_q_varphi_y_zc = q_varphi_y_zc.log_prob(y).view(k, x.shape[0], self.num_labeled).sum(dim=-1)
        log_q_varphi_phi_y_x = torch.logsumexp(log_q_varphi_y_zc, dim=0) - np.log(k)
        return log_q_varphi_phi_y_x

    def supervised_ELBO(self,
                        x: torch.tensor,
                        y: torch.tensor):
        batch_size = x.shape[0]

        # posterior parameters
        params_phi = self.encoder(x)

        # the approximate posterior
        q_phi_z_x = dist.Normal(*params_phi)
        z = q_phi_z_x.rsample()

        # image reconstruction
        r = self.decoder(z)

        # the first element of the vector z correspond to
        # the caracteristics we want to learn
        zc, zs = z.split([self.num_labeled, self.num_unlabeled], 1)

        # Classification
        logits_c = self.classifier(zc)
        # the label predicted distribution
        q_varphi_y_zc = dist.Bernoulli(logits=logits_c)
        log_q_varphi_y_zc = q_varphi_y_zc.log_prob(y).sum(dim=-1)

        # Conditional Prior
        mu_psi, sigma_psi = self.conditional_prior(y)
        params_psi = (
            torch.cat([mu_psi, torch.zeros(1, self.num_unlabeled, device=self.device).expand(batch_size, -1)], dim=1),
            torch.cat([sigma_psi, torch.ones(1, self.num_unlabeled, device=self.device).expand(batch_size, -1)], dim=1))
        p_psi_z_y = dist.Normal(*params_psi)

        # The prior labeled data
        p = self.y_prior_params.expand(batch_size, -1)
        log_py = dist.Bernoulli(p).log_prob(y).sum(dim=-1)

        # Classifier Loss
        log_q_varphi_phi_y_x = self.classifier_loss(x, y)

        # Generative model distribution
        p_theta_x_z = dist.Laplace(r, torch.ones_like(r))
        log_p_theta_x_z = p_theta_x_z.log_prob(x).sum(dim=(1,2,3))

        # Following appendix c.3.1
        q_phi_y_zc_ = dist.Bernoulli(logits=self.classifier(zc.detach()))
        log_q_phi_y_zc_ = q_phi_y_zc_.log_prob(y).sum(dim=-1)
        w = torch.exp(log_q_phi_y_zc_ - log_q_varphi_phi_y_x)

        # ELBO
        kl = dist.kl.kl_divergence(q_phi_z_x, p_psi_z_y).sum(dim=-1)
        elbo = (w * (log_p_theta_x_z - kl - log_q_varphi_y_zc) + log_py + log_q_varphi_phi_y_x).mean()
        return -elbo

    def unsupervised_ELBO(self,
                          images: torch.tensor):
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
        p_c = self.classifier(zc)
        # the label predicted distribution
        q_varphi_y_zc = dist.Bernoulli(logits=p_c)
        y = q_varphi_y_zc.sample()
        log_q_varphi_y_zc = q_varphi_y_zc.log_prob(y).sum(dim=-1)

        # Conditional Prior
        mu_psi, sigma_psi = self.conditional_prior(y)
        params_psi = (
            torch.cat([mu_psi, torch.zeros(1, self.num_unlabeled, device=self.device).expand(batch_size, -1)], dim=1),
            torch.cat([sigma_psi, torch.ones(1, self.num_unlabeled, device=self.device).expand(batch_size, -1)], dim=1))
        p_psi_z_y = dist.Normal(*params_psi)

        # The prior labeled data
        p = 0.5 * torch.ones_like(self.y_prior_params).expand(batch_size, -1)
        log_py = dist.Bernoulli(p).log_prob(y).sum(dim=-1)

        # Generative model distribution
        p_theta_x_z = dist.Laplace(r, torch.ones_like(r))
        log_p_theta_x_z = p_theta_x_z.log_prob(images).sum(dim=(1,2,3))

        # ELBO
        kl = dist.kl.kl_divergence(q_phi_z_x, p_psi_z_y).sum(dim=-1)
        elbo = (log_p_theta_x_z + log_py - kl - log_q_varphi_y_zc).mean()
        return -elbo

    def reconstruction(self, image):
        return self.decoder(dist.Normal(*self.encoder(image)).rsample())

    def conditional_generation(self, image, label, num_sample):
        z = dist.Normal(*self.encoder(image)).sample()
        _, zs = z.split([self.num_labeled, self.num_unlabeled], 1)
        zs = zs.expand([num_sample, -1])
        zc = dist.Normal(*self.conditional_prior(label)).sample([num_sample])
        new_z = torch.cat((zc, zs), axis=1)
        return self.decoder(new_z)


    def save(self, path:str):
        torch.save(self.encoder, os.path.join(path,'encoder.pt'))
        torch.save(self.decoder, os.path.join(path,'decoder.pt'))
        torch.save(self.classifier, os.path.join(path,'classifier.pt'))
        torch.save(self.conditional_prior, os.path.join(path,'conditional_prior.pt'))
    
    def load(self, path:str):
        self.encoder = torch.load(os.path.join(path,'encoder.pt'))
        self.decoder = torch.load(os.path.join(path,'decoder.pt'))
        self.classifier = torch.load(os.path.join(path,'classifier.pt'))
        self.conditional_prior = torch.load(os.path.join(path,'conditional_prior.pt'))