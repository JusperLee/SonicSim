import torch
from torch import nn


def linear():
    return nn.Identity()


def relu():
    return nn.ReLU()


def prelu():
    return nn.PReLU()


def leaky_relu():
    return nn.LeakyReLU()


def sigmoid():
    return nn.Sigmoid()


def softmax(dim=None):
    return nn.Softmax(dim=dim)


def tanh():
    return nn.Tanh()


def gelu():
    return nn.GELU()


def register_activation(custom_act):
    if (
        custom_act.__name__ in globals().keys()
        or custom_act.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Activation {custom_act.__name__} already exists. Choose another name."
        )
    globals().update({custom_act.__name__: custom_act})


def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError(
                "Could not interpret activation identifier: " + str(identifier)
            )
        return cls
    else:
        raise ValueError(
            "Could not interpret activation identifier: " + str(identifier)
        )


if __name__ == "__main__":
    print(globals().keys())
    print(globals().get("tanh"))
