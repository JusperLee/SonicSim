import json
from typing import Any, Dict, List, Optional, Tuple
import os
from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision("highest")
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
# from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import look2hear.system
import look2hear.datas
import look2hear.losses
from look2hear.utils import RankedLogger, instantiate, print_only
import warnings
warnings.filterwarnings("ignore")


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    
    # instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datas._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datas)
    
    # instantiate model
    print_only(f"Instantiating AudioNet <{cfg.model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)
    
    # instantiate optimizer
    print_only(f"Instantiating optimizer <{cfg.optimizer._target_}>")
    optimizer: torch.optim = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    # optimizer: torch.optim = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    # instantiate scheduler
    if cfg.get("scheduler"):
        print_only(f"Instantiating scheduler <{cfg.scheduler._target_}>")
        scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    else:
        scheduler = None
        
    # instantiate loss
    print_only(f"Instantiating loss <{cfg.loss._target_}>")
    loss: torch.nn.Module = hydra.utils.instantiate(cfg.loss)
    
    # instantiate metrics
    print_only(f"Instantiating metrics <{cfg.metrics._target_}>")
    metrics: torch.nn.Module = hydra.utils.instantiate(cfg.metrics)
    # instantiate system
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        model=model,
        loss_func=loss,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # instantiate callbacks
    callbacks: List[Callback] = []
    if cfg.get("early_stopping"):
        print_only(f"Instantiating early_stopping <{cfg.early_stopping._target_}>")
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))
    if cfg.get("checkpoint"):
        print_only(f"Instantiating checkpoint <{cfg.checkpoint._target_}>")
        checkpoint: pl.callbacks.ModelCheckpoint = hydra.utils.instantiate(cfg.checkpoint)
        callbacks.append(checkpoint)
        
    # instantiate logger
    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "logs"), exist_ok=True)
    logger = hydra.utils.instantiate(cfg.logger)
    
    # instantiate trainer
    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    
    trainer.fit(system, datamodule=datamodule)
    print_only("Training finished!")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(cfg.exp.dir, cfg.exp.name, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"))
    import wandb
    if wandb.run:
        print_only("Closing wandb!")
        wandb.finish()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="local/conf.yml",
        help="Full path to save best validation model",
    )
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    train(cfg)
    