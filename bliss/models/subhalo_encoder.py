# pylint: disable=R

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.distributions import LogNormal, Normal
from torch.optim import Adam

from bliss.catalog import TileCatalog, get_images_in_tiles
from bliss.models.encoder_layers import EncoderCNN, make_enc_final
from bliss.models.galaxy_encoder import CenterPaddedTilesTransform
from bliss.models.detection_encoder import (
    ConcatBackgroundTransform,
    EncoderCNN,
    LogBackgroundTransform,
    make_enc_final,
)


def get_subhalo_params_nll(true_params, params):
    """Get NLL of distributional parameters conditioned on the true galaxy parameters.

    Args:
        true_params:
            A tensor of the number of the true values of the galaxy parameters. Parameters
            much match the Galsim bulge-plus-disk galaxy parameterization,
            i.e. (total_flux, disk_frac, beta_radians, disk_q, a_d, bulge_q, a_b)
        params:
            A tensor of the number of the predicted values of the galaxy parameters (must
            also match Galsim bulge-plus-disk)

    Returns:
        A tensor value of the NLL (typically used as a loss to backpropagate)
    """
    assert true_params.shape[:-1] == params.shape[:-1]
    true_params = true_params.view(-1, true_params.shape[-1]).transpose(0, 1)
    params = params.view(-1, params.shape[-1]).transpose(0, 1)

    subhalo_R, subhalo_theta_R = true_params
    transformed_param_var_dist = [
        (subhalo_R, LogNormal),
        (subhalo_theta_R, LogNormal),
    ]

    # compute log-likelihoods of parameters and negate at end for NLL loss
    log_prob = torch.zeros(1, requires_grad=True).to(device=true_params.device)
    for i, (transformed_param, var_dist) in enumerate(transformed_param_var_dist):
        transformed_param_mean = params[2 * i]
        transformed_param_logvar = params[2 * i + 1]
        transformed_param_sd = (transformed_param_logvar.exp() + 1e-5).sqrt()

        parameterized_dist = var_dist(transformed_param_mean, transformed_param_sd)
        log_prob += parameterized_dist.log_prob(transformed_param).mean()

    return -log_prob


def sample_subhalo_encoder(var_dist_params, deterministic: Optional[bool]):
    """Sample from the encoded variational distribution.

    Args:
        var_dist_params: The output of `self.encode(image_ptiles)`,
            which is the distributional parameters in matrix form.

    Returns:
        A tensor with shape `n_samples * n_ptiles * max_sources * 7`
        consisting of Galsim bulge-plus-disk parameters.
    """

    subhalo_latent_dim = 2
    params_shape = list(var_dist_params[..., 0].shape) + [subhalo_latent_dim]
    subhalo_params = torch.zeros(params_shape)

    for latent_dim in range(subhalo_latent_dim):
        dist_mean = var_dist_params[..., 2 * latent_dim]
        dist_logvar = var_dist_params[..., 2 * latent_dim + 1]
        dist_sd = (dist_logvar.exp() + 1e-5).sqrt()
        dist = Normal(dist_mean, dist_sd)

        if deterministic is not None:
            param = dist.mean
        else:
            param = dist.rsample()
        positive_param_idxs = {0, 1}

        if latent_dim in positive_param_idxs:
            param = param.exp()
        subhalo_params[..., latent_dim] = param
    return subhalo_params


class SubhaloEncoder(pl.LightningModule):
    """Encodes the distribution of a latent variable representing a galaxy encoded with Galsim.

    This class implements the galaxy encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a latent variable
    representation of this image corresponding to the 7 parameters of a bulge-plus-disk
    Galsim galaxy (total_flux, disk_frac, beta_radians, disk_q, a_d, bulge_q, a_b).
    """

    def __init__(
        self,
        input_transform: Union[ConcatBackgroundTransform, LogBackgroundTransform],
        n_bands: int,
        tile_slen: int,
        ptile_slen: int,
        hidden: int,
        channel: int,
        spatial_dropout: float,
        dropout: float,
        optimizer_params: dict = None,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        self.input_transform = input_transform
        self.n_bands = n_bands
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        
        self.slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen
        self.optimizer_params = optimizer_params

        assert (ptile_slen - tile_slen) % 2 == 0
        self.border_padding = (ptile_slen - tile_slen) // 2

        # will be trained.
        latent_dim = 2
        dim_enc_conv_out = ((self.slen + 1) // 2 + 1) // 2
        n_bands_in = self.input_transform.output_channels(n_bands)
        self.enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self.enc_final = make_enc_final(
            channel * 4 * dim_enc_conv_out**2,
            hidden,
            2 * latent_dim,
            dropout,
        )

        # grid for center cropped tiles
        self.center_ptiles = CenterPaddedTilesTransform(self.tile_slen, self.ptile_slen)

        # consistency
        assert self.slen >= 20, "Cropped slen is not reasonable for average sized galaxies."

        if checkpoint_path is not None:
            self.load_state_dict(
                torch.load(Path(checkpoint_path), map_location=torch.device("cpu"))
            )

    def configure_optimizers(self):
        """Set up optimizers (pytorch-lightning method)."""
        return Adam(self.parameters(), **self.optimizer_params)

    def forward(self, image_ptiles, tile_locs):
        raise NotImplementedError("Please use encode()")

    def encode(self, image_ptiles: Tensor, tile_locs: Tensor) -> Tensor:
        n_samples, n_ptiles, max_sources, _ = tile_locs.shape
        centered_ptiles = self._get_images_in_centered_tiles(image_ptiles, tile_locs)
        assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.slen
        x = rearrange(centered_ptiles, "ns np c h w -> (ns np) c h w")
        enc_conv_output = self.enc_conv(x)
        subhalo_params_flat = self.enc_final(enc_conv_output)
        return rearrange(
            subhalo_params_flat,
            "(ns np ms) d -> ns np ms d",
            ns=n_samples,
            np=n_ptiles,
            ms=max_sources,
        )

    def sample(self, image_ptiles: Tensor, tile_locs: Tensor, deterministic: Optional[bool]):
        var_dist_params = self.encode(image_ptiles, tile_locs)
        subhalo_params = sample_subhalo_encoder(var_dist_params, deterministic)
        return subhalo_params.to(device=var_dist_params.device)

    def training_step(self, batch, batch_idx):
        """Pytorch lightning training step."""
        batch_size = len(batch["images"])
        loss = self._get_loss(batch)
        self.log("train/loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning validation step."""
        batch_size = len(batch["images"])
        pred = self._get_loss(batch)
        self.log("val/loss", pred["loss"], batch_size=batch_size)
        pred_out = {f"pred_{k}": v for k, v in pred.items()}
        return {**batch, **pred_out}

    def validation_epoch_end(self, outputs):
        """Pytorch lightning method run at end of validation epoch."""
        # put all outputs together into a single batch
        batch = {}
        for b in outputs:
            for k, v in b.items():
                if v.shape:
                    curr_val = batch.get(k, torch.tensor([], device=v.device))
                    batch[k] = torch.cat([curr_val, v])
        if self.n_bands == 1:
            self._make_plots(batch)

    def _get_loss(self, batch):
        images: Tensor = batch["images"]
        background: Tensor = batch["background"]
        tile_catalog = TileCatalog(
            self.tile_slen, {k: v for k, v in batch.items() if k not in {"images", "background"}}
        )

        image_ptiles = get_images_in_tiles(
            torch.cat((images, background), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        locs = rearrange(tile_catalog.locs, "n nth ntw ns hw -> 1 (n nth ntw) ns hw")
        subhalo_params_pred = self.encode(image_ptiles, locs)
        subhalo_params_pred = rearrange(
            subhalo_params_pred,
            "ns (n nth ntw) ms d -> (ns n) nth ntw ms d",
            ns=1,
            nth=tile_catalog.n_tiles_h,
            ntw=tile_catalog.n_tiles_w,
        )

        loss = get_subhalo_params_nll(
            batch["subhalo_params"],
            subhalo_params_pred,
        )

        return {
            "loss": loss,
        }

    def _get_images_in_centered_tiles(self, image_ptiles: Tensor, tile_locs: Tensor) -> Tensor:
        log_image_ptiles = self.input_transform(image_ptiles)
        assert log_image_ptiles.shape[-1] == log_image_ptiles.shape[-2] == self.ptile_slen
        # in each padded tile we need to center the corresponding galaxy/star
        return self.center_ptiles(log_image_ptiles, tile_locs)

    # pylint: disable=too-many-statements
    def _make_plots(self, batch, n_samples=5):
        pass

    def test_step(self, batch, batch_idx):
        pass
