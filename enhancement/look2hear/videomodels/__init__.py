###
# Author: Kai Li
# Date: 2021-06-21 12:04:03
# LastEditors: Please set LastEditors
# LastEditTime: 2022-03-21 01:48:53
###

from .wujian_videomodel import WujianVideoModel, update_wujian_parameter
from .resnet import ResNet, BasicBlock
from .resnet1D import ResNet1D, BasicBlock1D
from .shufflenetv2 import ShuffleNetV2
from .frcnn_videomodel import FRCNNVideoModel, update_frcnn_parameter
from .light_videomodel import LightVideomodel, update_light_parameter
from .resnet_videomodel import ResNetVideoModel, update_resnet_parameter

__all__ = [
    "WujianVideoModel",
    "update_wujian_parameter",
    "ResNet",
    "BasicBlock",
    "ResNet1D",
    "BasicBlock1D",
    "ShuffleNetV2",
    "FRCNNVideoModel",
    "update_frcnn_parameter",
    "LightVideomodel",
    "update_light_parameter",
    "ResNetVideoModel",
    "update_resnet_parameter",
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
