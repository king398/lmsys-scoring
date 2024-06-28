from sklearn.metrics import log_loss
import os
import random
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import yaml
import ast


def get_color_escape(r, g, b, background=False):
    return f'\033[{"48" if background else "38"};2;{r};{g};{b}m'


def seed_everything(seed=42):
    """Seed everything to ensure reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_score(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.softmax(dim=1).cpu().detach().numpy()
    score = log_loss(y_true, y_pred)

    return score


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # perform softmax on predictions
    predictions = torch.nn.functional.softmax(torch.tensor(predictions), dim=1).numpy()

    loss = log_loss(labels, predictions)
    results = {
        'log_loss': loss
    }
    return results


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


def write_yaml(config: dict, save_path: str) -> None:
    with open(save_path, 'w') as f:
        yaml.dump(config, f, )


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'linear_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('linear_head')
    if "p_layer" in lora_module_names:
        lora_module_names.remove("p_layer")
    return list(lora_module_names)


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []  # Handle cases where conversion fails
