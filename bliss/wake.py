import pytorch_lightning as pl
import torch
from torch import optim
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader


class WakeNet(pl.LightningModule):

    # ---------------
    # Model
    # ----------------

    def __init__(
        self,
        star_encoder,
        image_decoder,
        observed_img,
        hparams,
    ):
        super().__init__()

        self.star_encoder = star_encoder
        self.image_decoder = image_decoder
        self.image_decoder.requires_grad_(True)
        assert self.image_decoder.galaxy_decoder is None

        self.slen = image_decoder.slen
        self.border_padding = image_decoder.border_padding

        # observed image is batch_size (or 1) x n_bands x slen x slen
        self.padded_slen = self.slen + 2 * self.border_padding
        assert len(observed_img.shape) == 4
        assert observed_img.shape[-1] == self.padded_slen, "cached grid won't match."

        self.observed_img = observed_img

        # hyper-parameters
        self.save_hyperparameters(hparams)
        self.n_samples = self.hparams["n_samples"]
        self.lr = self.hparams["lr"]

        # get n_bands
        self.n_bands = self.image_decoder.n_bands

    def forward(self, obs_img):
        """Get reconstructed mean from running encoder and then decoder."""
        with torch.no_grad():
            self.star_encoder.eval()
            sample = self.star_encoder.sample_encoder(obs_img, self.n_samples)

        shape = sample["locs"].shape[:-1]
        zero_gal_params = torch.zeros(*shape, self.image_decoder.n_galaxy_params)
        recon_mean, _ = self.image_decoder.render_images(
            sample["n_sources"].contiguous(),
            sample["locs"].contiguous(),
            sample["galaxy_bool"].contiguous(),
            zero_gal_params,
            sample["fluxes"].contiguous(),
            add_noise=False,
        )

        return recon_mean

    # ---------------
    # Data
    # ----------------

    def train_dataloader(self):
        """Returns training dataloader (pytorch lightning)."""
        return DataLoader(self.observed_img, batch_size=None)

    def val_dataloader(self):
        """Returns validation dataloader (pytorch lightning)."""
        return DataLoader(self.observed_img, batch_size=None)

    # ---------------
    # Optimizer
    # ----------------

    def configure_optimizers(self):
        """Configures optimizers (pytorch lightning)."""
        return optim.Adam([{"params": self.image_decoder.parameters(), "lr": self.lr}])

    # ---------------
    # Training
    # ----------------

    def _get_loss(self, batch):
        img = batch.unsqueeze(0)
        recon_mean = self(img)
        error = -Normal(recon_mean, recon_mean.sqrt()).log_prob(img)

        image_indx_start = self.border_padding
        image_indx_end = self.border_padding + self.slen
        err = error[:, :, image_indx_start:image_indx_end, image_indx_start:image_indx_end]
        return err.sum((1, 2, 3)).mean()

    def training_step(self, batch, batch_idx):
        """Training step (pytorch lightning)."""
        loss = self._get_loss(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step (pytorch lightning)."""
        loss = self._get_loss(batch)
        self.log("val/loss", loss)
