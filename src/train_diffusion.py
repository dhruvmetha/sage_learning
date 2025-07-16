import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import hydra
import lightning.pytorch as pl
import torch
torch.set_float32_matmul_precision("medium")
from pathlib import Path

@hydra.main(config_path="../config", config_name="train_diffusion.yaml")
def main(cfg):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
        
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
    trainer.fit(model=model, datamodule=data_module)
    
if __name__ == "__main__":
    main()