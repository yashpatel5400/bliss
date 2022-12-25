# pylint: disable=R

from pathlib import Path
from typing import Optional, Union

from enum import Enum
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.distributions import VonMises, Normal
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

class IntervalType(Enum):
    OPEN = 1
    HALFOPEN = 2
    CLOSED = 3

def get_support_type(support):
    lower, upper = support
    if lower is None and upper is None:
        return IntervalType.OPEN
    elif lower is not None and upper is not None:
        return IntervalType.CLOSED
    return IntervalType.HALFOPEN

class ParamsEncoder(pl.LightningModule):
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
        param_supports,
        params_tag,
        params_filter_tag,
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

        self.param_supports = param_supports
        self.params_tag = params_tag
        self.params_filter_tag = params_filter_tag
        
        # will be trained.
        self.latent_dim = len(self.param_supports)
        dim_enc_conv_out = ((self.slen + 1) // 2 + 1) // 2
        n_bands_in = self.input_transform.output_channels(n_bands)
        self.enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self.enc_final = make_enc_final(
            channel * 4 * dim_enc_conv_out**2,
            hidden,
            2 * self.latent_dim,
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
        params_flat = self.enc_final(enc_conv_output)
        return rearrange(
            params_flat,
            "(ns np ms) d -> ns np ms d",
            ns=n_samples,
            np=n_ptiles,
            ms=max_sources,
        )

    def sample(self, image_ptiles: Tensor, tile_locs: Tensor, deterministic: Optional[bool]):
        var_dist_params = self.encode(image_ptiles, tile_locs)
        params = self.sample_encoder(var_dist_params, deterministic)
        return params.to(device=var_dist_params.device)

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

    def _get_loss(self, batch, is_val=False):
        images: Tensor = batch["images"]
        background: Tensor = batch["background"]
        tile_catalog = TileCatalog(
            self.tile_slen, {k: v for k, v in batch.items() if k not in {"images", "background", "global"}}
        )

        image_ptiles = get_images_in_tiles(
            torch.cat((images, background), dim=1),
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        locs = rearrange(tile_catalog.locs, "n nth ntw ns hw -> 1 (n nth ntw) ns hw")
        params_pred = self.encode(image_ptiles, locs)
        params_pred = rearrange(
            params_pred,
            "ns (n nth ntw) ms d -> (ns n) nth ntw ms d",
            ns=1,
            nth=tile_catalog.n_tiles_h,
            ntw=tile_catalog.n_tiles_w,
        )

        if self.params_filter_tag is not None:
            params_filter = batch[self.params_filter_tag]
        else:
            params_filter = None
        loss = self.get_params_nll(
            params_filter,
            batch[self.params_tag],
            params_pred,
            is_val=is_val,
        )

        return {
            "loss": loss,
        }

    def get_params_nll(self, params_filter, true_params, var_dist_params, is_val=False):
        assert true_params.shape[:-1] == var_dist_params.shape[:-1]
        true_params = true_params.view(-1, true_params.shape[-1]).transpose(0, 1)
        var_dist_params = var_dist_params.view(-1, var_dist_params.shape[-1]).transpose(0, 1)
        
        if params_filter is not None:
            params_filter = params_filter.view(-1)
            true_params = true_params[:, params_filter > 0]
            var_dist_params = var_dist_params[:, params_filter > 0]

        # compute log-likelihoods of parameters and negate at end for NLL loss
        log_prob = torch.zeros(1, requires_grad=True).to(device=true_params.device)
        for i, param_support in enumerate(self.param_supports):
            param_support_type = get_support_type(param_support)
            transformed_param_mean = var_dist_params[2 * i]
            transformed_param_logvar = var_dist_params[2 * i + 1]
            transformed_param_sd = (transformed_param_logvar.exp() + 1e-5).sqrt()

            angle_param = False # param_support == (0, 2 * np.pi) # use VonMises variational distribution if angle parameter

            param = true_params[i]
            if param_support_type == IntervalType.OPEN or angle_param:
                transformed_param = param
            elif param_support_type == IntervalType.HALFOPEN:
                transformed_param = torch.log(param - param_support[0])
            elif param_support_type == IntervalType.CLOSED:
                transformed_param = torch.logit((param - param_support[0]) / (param_support[1] - param_support[0]))
            
            if angle_param:
                parameterized_dist = VonMises(transformed_param_mean, transformed_param_sd)
            else:
                parameterized_dist = Normal(transformed_param_mean, transformed_param_sd)
            log_prob_i = parameterized_dist.log_prob(transformed_param).mean()
            log_prob += log_prob_i

            if is_val:
                print(f"{i} -> {log_prob_i}")

        return -log_prob


    def sample_encoder(self, var_dist_params, deterministic: Optional[bool]):
        params_shape = list(var_dist_params[..., 0].shape) + [self.latent_dim]
        params = torch.zeros(params_shape)

        for i, param_support in enumerate(self.param_supports):
            angle_param = False # param_support == (0, 2 * np.pi) # use VonMises variational distribution if angle parameter
            
            dist_mean = var_dist_params[..., 2 * i]
            dist_logvar = var_dist_params[..., 2 * i + 1]
            dist_sd = (dist_logvar.exp() + 1e-5).sqrt()
            
            if angle_param:
                dist = VonMises(dist_mean, dist_sd)
            else:
                dist = Normal(dist_mean, dist_sd)

            if deterministic:
                transformed_param = dist.mean
            else:
                transformed_param = dist.sample()
            
            param_support_type = get_support_type(param_support)
            if param_support_type == IntervalType.OPEN:
                param = transformed_param
            elif param_support_type == IntervalType.HALFOPEN:
                param = transformed_param.exp() + param_support[0]
            elif param_support_type == IntervalType.CLOSED:
                if angle_param:
                    param = transformed_param + np.pi
                else:
                    param = torch.sigmoid(transformed_param) * (param_support[1] - param_support[0]) + param_support[0]
            params[..., i] = param
        return params

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
