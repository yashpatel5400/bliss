def test_galaxy_autoencoder_toy_gaussian(model_setup, devices):
    use_cuda = devices.use_cuda
    overrides = {
        "model": "galaxy_net",
        "+model.residual_delay_n_steps": 50 if use_cuda else 0,
        "dataset": "toy_gaussian",
        "dataset.batch_size": 64 if use_cuda else 10,
        "dataset.n_batches": 10 if use_cuda else 1,
        "training": "unittest",
        "training.n_epochs": 101 if use_cuda else 2,
        "training.trainer.check_val_every_n_epoch": 50 if use_cuda else 1,
        "optimizer": "adam",
    }

    # train galaxy_vae
    galaxy_ae = model_setup.get_trained_model(overrides)
    results = model_setup.test_model(overrides, galaxy_ae)

    # only expect tests to pass in cuda:
    if not devices.use_cuda:
        return

    assert results["max_residual"] < 25


def test_galaxy_autoencoder_bulge_disk(model_setup, devices):
    use_cuda = devices.use_cuda
    overrides = {
        "model": "galaxy_net",
        "dataset": "sdss_galaxies",
        "dataset.batch_size": 128 if use_cuda else 10,
        "dataset.n_batches": 5 if use_cuda else 1,
        "dataset.num_workers": 10 if use_cuda else 0,
        "training": "unittest",
        "training.n_epochs": 101 if use_cuda else 2,
        "optimizer": "adam",
    }

    # train galaxy_vae
    galaxy_ae = model_setup.get_trained_model(overrides)
    results = model_setup.test_model(overrides, galaxy_ae)

    # only expect tests to pass in cuda:
    if not devices.use_cuda:
        return

    assert results["max_residual"] < 30
