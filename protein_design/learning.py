"""
Custom optimizers and learning rate schedulers
"""

# adapted from https://stackoverflow.com/a/66212699
class WarmupAnnealLR:
    def __init__(self, optimizer, warmup_steps: int = 1000) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.iter = 1

    def get_rate(self):
        return 128 ** (-0.5) * min(self.iter ** (-0.5), self.iter * self.warmup_steps ** (-1.5))

    def step(self):
        for p in self.optimizer.param_groups:
            p['lr'] = self.get_rate()
        self.iter += 1
