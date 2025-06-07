from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from functools import partial
import hydra

class WarmupCosineScheduler:
    """
    Combines a Linear warm-up (end_lr -> base_lr) with
    a CosineAnnealing decay (base_lr -> end_lr).
    """
    def __init__(self, optimizer, schedulers_cfg, milestones):
        
        self.optimizer = optimizer
        inner = []
        for sch in schedulers_cfg:
            if isinstance(sch, partial):
                # Hydra already gave you a partial factory
                print(sch)
                inner.append(sch(optimizer=optimizer))
            else:
                # it’s a DictConfig, so instantiate it
                inner.append(hydra.utils.instantiate(sch, optimizer=optimizer))
        
        self.scheduler = SequentialLR(
            optimizer,
            schedulers=inner,
            milestones=[milestones],
        )

    def step(self, *args, **kwargs):
        return self.scheduler.step(*args, **kwargs)

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, sd):
        self.scheduler.load_state_dict(sd)
