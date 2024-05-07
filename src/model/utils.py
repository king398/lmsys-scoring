from sklearn.metrics import log_loss
import os
import random
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import yaml


def get_color_escape(r, g, b, background=False):
    return f'\033[{"48" if background else "38"};2;{r};{g};{b}m'


def seed_everything(seed=20):
    """Seed everything to ensure reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(folder):
    filename = os.path.join(folder, 'train')
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler = FileHandler(filename=f"{filename}.log")
    handler.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def get_score(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.softmax(dim=1).cpu().detach().numpy()
    score = log_loss(y_true, y_pred)

    return score


def get_scheduler(scheduler_name, warmup_steps, optimizer, num_train_steps, num_cycles=1):
    if scheduler_name == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=num_train_steps
        )
    elif scheduler_name == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=num_train_steps, num_cycles=num_cycles,
        )
    return scheduler


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def write_yaml(config: dict, save_path: str) -> None:
    with open(save_path, 'w') as f:
        yaml.dump(config, f, )
