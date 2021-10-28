import numpy as np


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))
