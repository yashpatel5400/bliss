import sys
import copy

sys.path.append("../../")

from einops import rearrange
from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources
from bliss import reporting
from bliss.encoder import Encoder
from bliss.inference import SDSSFrame
from bliss.datasets import sdss
from bliss.inference import reconstruct_scene_at_coordinates
from case_studies.substructure.plots.main import load_models

import matplotlib.pyplot as plt

plt.style.use("ggplot")
import torch

from astropy.table import Table
import plotly.express as px
import plotly.graph_objects as go
from hydra import compose, initialize
from hydra.utils import instantiate
import numpy as np

with initialize(config_path="config"):
    cfg = compose("config", overrides=[])

device = torch.device("cuda:0")
galaxy_encoder = instantiate(cfg.models.galaxy_encoder).to(device).eval()
galaxy_encoder.load_state_dict(torch.load(cfg.plots.galaxy_checkpoint, map_location=galaxy_encoder.device))

lens_encoder = instantiate(cfg.models.lens_encoder).to(device).eval()
lens_encoder.load_state_dict(torch.load(cfg.plots.lens_checkpoint, map_location=lens_encoder.device))

torch.cuda.empty_cache()

dataset = instantiate(
    cfg.datasets.simulated,
    generate_device="cuda:0",
)

total_params = 0
batch_size = 1
trials = 10#_000
num_samples = 100

coverage_type_to_encoder = {
    "galaxy": galaxy_encoder,
    "lens": lens_encoder,
}

num_galaxy_params = 7
num_lens_params = 8
coverage_type_to_result = {
    "galaxy": np.zeros(num_galaxy_params),
    "lens": np.zeros(num_lens_params),
}

for _ in range(trials):
    tile_catalog = dataset.sample_prior(batch_size, cfg.datasets.simulated.n_tiles_h, cfg.datasets.simulated.n_tiles_w)
    packed_images, backgrounds = dataset.simulate_image_from_catalog(tile_catalog)
    images = packed_images["global"]

    full_true = tile_catalog["global"].cpu().to_full_params()
    image_ptiles = get_images_in_tiles(
        torch.cat((images, backgrounds), dim=1),
        80,
        80,
    )
    image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
    locs = rearrange(tile_catalog["global"].locs, "n nth ntw ns hw -> 1 (n nth ntw) ns hw")

    for coverage_type in coverage_type_to_encoder:
        if coverage_type == "lens":
            true_params = full_true["lens_params"].cpu().numpy()[0, ...][:, :num_lens_params]
        else:
            true_params = full_true["galaxy_params"].cpu().numpy()[0, ...][:, :num_galaxy_params]

        samples = []
        for _ in range(num_samples):
            params = coverage_type_to_encoder[coverage_type].sample(image_ptiles, locs, deterministic=False)
            samples.append(params.detach().cpu().numpy())
            
        samples = np.array(samples)

        confidence_percent = 0.90
        alpha = ((1 - confidence_percent) / 2) * 100
        lower_ci = np.percentile(samples, alpha, axis=0)
        upper_ci = np.percentile(samples, 100 - alpha, axis=0)

        contained = np.logical_and(lower_ci <= true_params, true_params <= upper_ci).astype("int")
        coverage_type_to_result[coverage_type] += np.sum(contained, axis=0).squeeze()
        
for coverage_type in coverage_type_to_result:
    print(f"{coverage_type} : {coverage_type_to_result[coverage_type] / trials}")
