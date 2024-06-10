import argparse
import yaml
from pathlib import Path
from utils import seed_everything, write_yaml
import warnings
from dataset import WinnerDataset
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from train_fn import oof_fn
from sklearn.metrics import log_loss
import numpy as np
from model import WinnerModel
from torch import nn
def main(cfg):
    warnings.filterwarnings("ignore")
    seed_everything(cfg['seed'])
    df = pd.read_csv(cfg['train_csv'])
    predictions_all = []
    targets_all = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in cfg['folds']:
        valid_df = df[df['fold'] == i].reset_index(drop=True)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
        valid_dataset = WinnerDataset(df=valid_df, max_len=cfg['max_len'], tokenizer=tokenizer)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'] * 2 , shuffle=False,
                                  num_workers=cfg['num_workers'], collate_fn=DataCollatorWithPadding(tokenizer))
        model = WinnerModel(cfg['model_name'], cfg['num_classes'], tokenizer)
        model.load_state_dict(torch.load(f"{cfg['model_dir']}/fold_{i}.pth"))
        model = model.to(device)
        predictions, targets = oof_fn(valid_loader, model, device)
        predictions_all.extend(predictions)
        targets_all.extend(targets)
        score = log_loss(targets, predictions, )
        print(f"Score for Fold {i}: {score}")
    score = log_loss(targets_all, predictions_all)
    print(f"Overall Score: {score}")
    cfg['oof_score'] = float(score)
    write_yaml(cfg, f"{cfg['model_dir']}/config.yaml")
    np.save(f"{cfg['model_dir']}/predictions.npy", predictions_all)
    np.save(f"{cfg['model_dir']}/targets.npy", targets_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)
