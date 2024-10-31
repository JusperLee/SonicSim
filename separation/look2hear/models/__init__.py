from .tdanet import TDANet
from .afrcnn import AFRCNN
from .sudormrf import SuDORMRF
from .ConvTasnet import ConvTasNet
from .dprnn import DPRNNTasNet
from .TFGNet import TFGridNet
from .bsrnn import BSRNN
from .dptnet import DPTNetModel
from .mossformer import MossFormer
from .mossformer2 import MossFormer2
from .skim import SkiMNet

__all__ = [
    "SkiMNet",
    "TDANet",
    "AFRCNN",
    "ConvTasNet",
    "SuDORMRF",
    "DPRNNTasNet",
    "TFGridNet",
    "BSRNN",
    "DPTNetModel",
    "MossFormer",
    "MossFormer2",
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
