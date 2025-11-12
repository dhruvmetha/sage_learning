import os
import sys
import hydra
import lightning.pytorch as pl
import torch
torch.set_default_dtype(torch.float32)

# Configure TF32 for faster computation on Ampere+ GPUs (RTX A4000)
# Using old API only - PyTorch Lightning is not yet compatible with new API
torch.set_float32_matmul_precision('medium')  # Enables TF32 for matmul operations

# Use file_system sharing strategy to avoid NFS issues with multiprocessing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from pathlib import Path

# Add project root to Python path to ensure imports work
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

@hydra.main(config_path="../config", config_name="train_diffusion.yaml", version_base="1.1")
def main(cfg):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
    
    # Don't set CUDA_VISIBLE_DEVICES for multi-GPU training
    # Lightning's DDP strategy handles device selection automatically
        
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.get("run_path", None) is not None: # for example: +run_path=../../../outputs/2025-07-11/02-02-32
        run_path = Path(cfg.run_path)
        print(f"Loading model from {run_path}, {run_path.parent}")
        checkpoint_dir = run_path / "checkpoints"
        print(f"Checkpoint directory: {checkpoint_dir}")
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        print(f"Found {len(checkpoint_files)} checkpoints")
        checkpoint_path = ""
        for checkpoint_file in checkpoint_files:
            if "epoch" in checkpoint_file.name:
                checkpoint_path = checkpoint_file
                break
        if checkpoint_path == "":
            raise ValueError("No checkpoint found")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"✅ Model loaded successfully: {type(model).__name__}")
    
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Check if resuming from checkpoint
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        print(f"✅ Resuming training from checkpoint: {ckpt_path}")
    
    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)
    
if __name__ == "__main__":
    main()