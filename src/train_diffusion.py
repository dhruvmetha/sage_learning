import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import hydra
import lightning.pytorch as pl

@hydra.main(config_path="../config", config_name="train_diffusion.yaml")
def main(cfg):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)
        
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    trainer.fit(model=model, datamodule=data_module)
    
if __name__ == "__main__":
    main()