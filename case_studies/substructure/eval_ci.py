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
torch.cuda.empty_cache()

dataset = instantiate(
    cfg.datasets.simulated,
    generate_device=device,
)

galaxy = instantiate(cfg.models.galaxy_encoder).to(device).eval()
galaxy.load_state_dict(torch.load(cfg.plots.galaxy_checkpoint, map_location=galaxy.device))

lens = instantiate(cfg.models.lens_encoder).to(device).eval()
lens.load_state_dict(torch.load(cfg.plots.lens_checkpoint, map_location=lens.device))

contained = [np.zeros(7), np.zeros(12)]

print_freq = 50
batch_size = 1
trials = 10_000
num_samples = 100

for trial in range(trials):
    tile_catalogs = {}
    tile_catalogs["main_deflector"] = dataset.image_prior.sample_prior(dataset.tile_slen, batch_size, dataset.n_tiles_h, dataset.n_tiles_w)
    if dataset.substructure_prior:
        tile_catalogs["substructure"] = dataset.substructure_prior.sample_prior(dataset.substructure_tile_slen, batch_size, dataset.substructure_n_tiles_h, dataset.substructure_n_tiles_w)
    images, backgrounds = dataset.simulate_image_from_catalog(tile_catalogs)

    full_true = tile_catalogs["main_deflector"].to_full_params()
    img_bg = torch.cat((images[0], backgrounds[0]), dim=0).unsqueeze((0)).to(device)

    galaxy_samples = []
    lens_samples = []
    for _ in range(num_samples):
        galaxy_samples.append(galaxy.sample(img_bg, tile_catalogs["main_deflector"].locs[0], deterministic=False).cpu().numpy())
        lens_samples.append(lens.sample(img_bg, tile_catalogs["main_deflector"].locs[0], deterministic=False).cpu().numpy())

    for i, (samples_list, truth) in enumerate(zip((galaxy_samples, lens_samples), (full_true["galaxy_params"], full_true["lens_params"]))):
        samples = np.array(samples_list)
        truth = truth.cpu().numpy()

        confidence_percent = 0.90
        alpha = ((1 - confidence_percent) / 2) * 100
        lower_ci = np.percentile(samples, alpha, axis=0)
        upper_ci = np.percentile(samples, 100 - alpha, axis=0)

        contained_trial = np.logical_and(lower_ci <= truth, truth <= upper_ci).astype("int")
        contained[i] += np.sum(contained_trial, axis=0).squeeze()

    if trial % print_freq == 0:
        for enc_type, total_contained in zip(("galaxy", "lens"), contained):
            print(f"[Trial {trial}] {enc_type} : {total_contained / (trial + 1)}")
