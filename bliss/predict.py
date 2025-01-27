import torch
from hydra.utils import instantiate


def prepare_image(x, device):
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device=device)
    # image dimensions must be a multiple of 16
    height = x.size(2) - (x.size(2) % 16)
    width = x.size(3) - (x.size(3) % 16)
    return x[:, :, :height, :width]


def predict(cfg):
    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    encoder.eval()

    sdss = instantiate(cfg.predict.dataset)
    batch = {
        "images": prepare_image(sdss[0]["image"], cfg.predict.device),
        "background": prepare_image(sdss[0]["background"], cfg.predict.device),
    }

    with torch.no_grad():
        pred = encoder.encode_batch(batch)
        est_cat = encoder.variational_mode(pred)

    print("{} light sources detected".format(est_cat.n_sources.item()))
