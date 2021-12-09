import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, IterableDataset

from torch.distributions import Poisson

from bliss.models.decoder import ImageDecoder


class SimulatedStarnetDataset(pl.LightningDataModule, IterableDataset):
    def __init__(
        self, 
        decoder_kwargs,
        n_batches=10,
        batch_size=32,
        generate_device="cpu",
        testing_file=None, 
        mean_background_vals = [686., 1123.], 
        background_sd = 100, 
    ):
        super().__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.image_decoder = ImageDecoder(**decoder_kwargs).to(generate_device)
        self.image_decoder.requires_grad_(False)  # freeze decoder weights.
        self.testing_file = testing_file

        # check sleep training will work.
        n_tiles_per_image = self.image_decoder.n_tiles_per_image
        total_ptiles = n_tiles_per_image * self.batch_size
        assert total_ptiles > 1, "Need at least 2 tiles over all batches."
        
        # custom backgrounds 
        self.image_decoder.background_values = [0] * len(mean_background_vals) 
        self.mean_background_vals = mean_background_vals
        self.background_sd = background_sd 
        
        # we overwrote these methods
#         self.image_decoder._sample_n_sources = self._sample_n_sources
        
        
    def __iter__(self):
        return self.batch_generator()

    def batch_generator(self):
        for _ in range(self.n_batches):
            yield self.get_batch()

    def sample_backgrounds(self, slen, batch_size): 
        
        nbands = len(self.mean_background_vals)
        mu = torch.Tensor(self.mean_background_vals).to(self.image_decoder.device).unsqueeze(0)
        sd = self.background_sd
        
        normal_samples = torch.randn(size = (batch_size, nbands)).to(self.image_decoder.device)
        
        sampled_background_vals = mu + normal_samples * sd
        
        return sampled_background_vals
    
#     def _sample_n_sources(self, batch_size=1):
        
#         # this overwrites the `_sample_n_sources` method in the image decoder
        
#         # sample number of sources
#         # first sample poisson prior parameter
#         pois_prior_lambda = torch.rand(batch_size).to(self.image_decoder.device) * 0.6 + 0.4
#         poisson = Poisson(pois_prior_lambda)
#         n_sources = poisson.sample((self.image_decoder.n_tiles_per_image, )).transpose(0, 1)
        
#         # long() here is necessary because used for indexing and one_hot encoding.
#         n_sources = n_sources.clamp(max=self.image_decoder.max_sources,
#                                     min=self.image_decoder.min_sources)
        
#         return n_sources.long()

    def get_batch(self):
        with torch.no_grad():
            batch = self.image_decoder.sample_prior(batch_size=self.batch_size)
            images, _ = self.image_decoder.render_images(
                batch["n_sources"],
                batch["locs"],
                batch["galaxy_bool"],
                batch["galaxy_params"],
                batch["fluxes"],
                add_noise=False,
            )
            
            
            # add in the random background
            background = self.sample_backgrounds(images.shape[-1], self.batch_size)
            images = images + background.unsqueeze(-1).unsqueeze(-1)
            
            images = self.image_decoder._apply_noise(images)
            
            batch.update(
                {
                    "images": images,
                    "slen": torch.tensor([self.image_decoder.slen]),
                }
            )

        return batch

    def train_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self, batch_size=None, num_workers=0)

    def test_dataloader(self):
        dl = DataLoader(self, batch_size=None, num_workers=0)

        if self.testing_file:
            test_dataset = BlissDataset(self.testing_file)
            dl = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)

        return dl

