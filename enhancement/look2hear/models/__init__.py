###
# Author: Kai Li
# Date: 2022-02-12 15:16:35
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-10-04 16:24:53
###
from .fullband import Fullband
from .fullsubnet import FullSubnet
from .fastfullsubnet import FastFullSubnet
from .dccrn import DCCRN
from .fullsubnet_plus import FullSubNet_Plus
from .taylorsenet import TaylorSENet
from .gagnet import GaGNet
from .g2net import G2Net
from .inter_subnet import Inter_SubNet
from .bsrnn_espnet import BSRNNESPNet
from .frcrn import FRCRN
from .sudormrf import SuDORMRF

__all__ = [
    "Fullband",
    "FullSubnet",
    "FastFullSubnet",
    "DCCRN",
    "FullSubNet_Plus",
    "TaylorSENet",
    "GaGNet",
    "G2Net",
    "Inter_SubNet",
    "BSRNNESPNet",
    "FRCRN",
    "SuDORMRF"
]


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if (
        custom_model.__name__ in globals().keys()
        or custom_model.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Model {custom_model.__name__} already exists. Choose another name."
        )
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
