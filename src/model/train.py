import argparse
import os
import yaml
from pathlib import Path
from utils import seed_everything, get_logger, write_yaml
import warnings
import gc
from accelerate import Accelerator, DistributedDataParallelKwargs
from dataset import WinnerDataset
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from model import WinnerModel
from torch.optim import AdamW
from utils import get_optimizer_params, get_scheduler
import numpy as np
from torch.nn import CrossEntropyLoss
from train_fn import train_fn, valid_fn
from tokenizers import AddedToken


def main(cfg):
    warnings.filterwarnings("ignore")
    seed_everything(cfg['seed'])

    gc.enable()
    accelerator = Accelerator(mixed_precision=cfg['mixed_precision'], log_with=['wandb'], kwargs_handlers=[
        DistributedDataParallelKwargs(gradient_as_bucket_view=True, find_unused_parameters=True)])
    accelerator.init_trackers(project_name="lmsys-winner", config=cfg)
    df = pd.read_csv(cfg['train_csv'])
    for i in cfg['folds']:
        accelerator.print(f"Training Fold: {i}")
        train_df = df[df['fold'] != i].reset_index(drop=True)
        valid_df = df[df['fold'] == i].reset_index(drop=True)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
        train_dataset = WinnerDataset(df=train_df, max_len=cfg['max_len'], tokenizer=tokenizer)
        valid_dataset = WinnerDataset(df=valid_df, max_len=cfg['max_len'], tokenizer=tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                                  num_workers=cfg['num_workers'], collate_fn=DataCollatorWithPadding(tokenizer))
        valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'] * 2, shuffle=False,
                                  num_workers=cfg['num_workers'], collate_fn=DataCollatorWithPadding(tokenizer))

        model = WinnerModel(cfg['model_name'], cfg['num_classes'], tokenizer)
        optimizer = AdamW(model.parameters(), lr=float(cfg['lr']))
        scheduler = get_scheduler(cfg['scheduler'], cfg['warmup_steps'], optimizer, len(train_loader) * cfg['epochs'])
        criterion = CrossEntropyLoss()
        model, optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                                      train_loader, valid_loader)
        best_score = np.inf
        for epoch in range(cfg['epochs']):
            train_fn(train_loader, model, optimizer, scheduler, accelerator, i, epoch, criterion)
            score = valid_fn(valid_loader, model, accelerator, i, epoch, criterion)
            if score < best_score:
                best_score = score
                accelerator.save(accelerator.unwrap_model(model).state_dict(), f"{cfg['model_dir']}/fold_{i}.pth")
        accelerator.print(f"Best Score: {best_score} for Fold: {i}")
    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(cfg['model_dir'], exist_ok=True)
    write_yaml(cfg, save_path=f"{cfg['model_dir']}/config.yaml")
    main(cfg)
