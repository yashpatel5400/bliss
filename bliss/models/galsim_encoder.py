# pylint: disable=R

from typing import Optional, Union
import numpy as np

from bliss.models.detection_encoder import (
    ConcatBackgroundTransform,
    LogBackgroundTransform,
)
from bliss.models.params_encoder import ParamsEncoder

class GalsimEncoder(ParamsEncoder):
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
        # bulge-plus-disk for Galsim inference
        param_supports = [
            (0, None), # total_flux
            (0, 1), # disk_frac
            (0, 2 * np.pi), # beta_radians
            (0, 1), # disk_q
            (0, None), # a_d
            (0, 1), # bulge_q
            (0, None), # a_b
        ]

        super().__init__(
            input_transform=input_transform,
            n_bands=n_bands,
            tile_slen=tile_slen,
            ptile_slen=ptile_slen,
            hidden=hidden,
            channel=channel,
            spatial_dropout=spatial_dropout,
            dropout=dropout,
            optimizer_params=optimizer_params,
            checkpoint_path=checkpoint_path,
            param_supports=param_supports,
            params_tag="galaxy_params",
            params_filter_tag="galaxy_bools",
        )