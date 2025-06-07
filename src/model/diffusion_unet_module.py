import torch
import torch.nn.functional as F
import torchvision
import os
import lightning.pytorch as pl

class DiffusionUNetModule(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def forward(self, x):
        return self.model(x)