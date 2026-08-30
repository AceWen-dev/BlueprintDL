from torch.optim import Adam, AdamW, SGD

from dlkit.registry import OPTIMIZERS


@OPTIMIZERS.register()
class Adam(Adam):
    pass


@OPTIMIZERS.register()
class AdamW(AdamW):
    pass


@OPTIMIZERS.register()
class SGD(SGD):
    pass
