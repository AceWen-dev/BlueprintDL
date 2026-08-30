from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR

from dlkit.registry import SCHEDULERS


@SCHEDULERS.register()
class PolyLR:
    per_iter = True

    def __init__(self, optimizer, max_iters, power=0.9, last_iter=-1):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.power = power
        self.last_iter = last_iter
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.last_iter += 1
        factor = (1.0 - self.last_iter / float(self.max_iters)) ** self.power
        factor = max(0.0, factor)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base * factor

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {'last_iter': self.last_iter, 'base_lrs': self.base_lrs}

    def load_state_dict(self, state):
        self.last_iter = state['last_iter']
        self.base_lrs = state['base_lrs']


@SCHEDULERS.register()
class StepLR(StepLR):
    pass


@SCHEDULERS.register()
class MultiStepLR(MultiStepLR):
    pass


@SCHEDULERS.register()
class CosineAnnealingLR(CosineAnnealingLR):
    pass
