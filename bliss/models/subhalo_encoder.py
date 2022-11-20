# pylint: disable=R

from typing import Optional, Union

from bliss.models.detection_encoder import (
    ConcatBackgroundTransform,
    LogBackgroundTransform,
)
from bliss.models.params_encoder import ParamsEncoder

class SubhaloEncoder(ParamsEncoder):
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
        precentered: Optional[bool] = False,
    ):
        param_supports = [
            (0, None), # subhalo_R
            (0, None), # subhalo_theta_R
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
            precentered=precentered,
            param_supports=param_supports,
            params_tag="subhalo_params",
            params_filter_tag=None,
        )