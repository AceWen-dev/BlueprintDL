class EarlyStopping:
    def __init__(self, metric='val_loss', mode='min', patience=10, min_delta=0.0):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def attach(self, trainer, event='after_epoch'):
        trainer.register_hook(event, self._on_event)

    def _on_event(self, trainer, metrics=None, **kwargs):
        if not metrics or self.metric not in metrics:
            return
        current = metrics[self.metric]
        if self.best is None:
            self.best = current
            return
        improved = (
            current > self.best + self.min_delta
            if self.mode == 'max'
            else current < self.best - self.min_delta
        )
        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.stop_training = True
                trainer.logger.info(
                    'EarlyStopping: no improvement for %d epochs (%s=%.4f)',
                    self.patience, self.metric, self.best,
                )


class LRSchedulerOnPlateau:
    def __init__(self, metric='val_loss', mode='min', factor=0.1, patience=5):
        self.metric = metric
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.best = None
        self.counter = 0

    def attach(self, trainer, event='after_epoch'):
        trainer.register_hook(event, self._on_event)

    def _on_event(self, trainer, metrics=None, **kwargs):
        if not metrics or self.metric not in metrics:
            return
        current = metrics[self.metric]
        if self.best is None:
            self.best = current
            return
        improved = current > self.best if self.mode == 'max' else current < self.best
        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                for group in trainer.optimizer.param_groups:
                    group['lr'] *= self.factor
                trainer.logger.info('Reduced LR to %.2e', trainer.optimizer.param_groups[0]['lr'])
