from hydra import compose, initialize

from bliss.datasets.galsim_galaxies import SavedGalaxies, SDSSCatalogGalaxies


def test_catalog_galaxies():
    # just a smoke test
    overrides = {"dataset": "sdss_catalog_galaxies", "dataset.kwargs.batch_size": 128}
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        ds = SDSSCatalogGalaxies(**cfg.dataset.kwargs)
        for _ in ds.train_dataloader():
            pass
        for _ in ds.val_dataloader():
            pass
        for _ in ds.test_dataloader():
            pass


def test_saved_galaxies():
    # just a smoke test
    overrides = {"dataset": "saved_galaxies", "dataset.kwargs.batch_size": 64}
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        ds = SavedGalaxies(**cfg.dataset.kwargs)
        for _ in ds.train_dataloader():
            pass
        for _ in ds.val_dataloader():
            pass
        for _ in ds.test_dataloader():
            pass
