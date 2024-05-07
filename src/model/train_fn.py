import torch
from tqdm import tqdm
from utils import get_score, get_color_escape
from accelerate import Accelerator

tqdm_color = get_color_escape(0, 229, 255)  # Red color for example
tqdm_style = {
    'bar_format': f'{tqdm_color}{{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}},'
                  f' {{rate_fmt}}{{postfix}}]{get_color_escape(255, 255, 255)}'}


def train_fn(
        train_loader,
        model,
        optimizer,
        scheduler,
        accelerator: Accelerator,
        fold,
        epoch,
        criterion,

):
    model.train()
    total_loss = 0
    stream = tqdm(train_loader, disable=not accelerator.is_local_main_process, **tqdm_style)
    for i, batch in enumerate(stream):
        targets = batch.pop('targets')
        output = model(batch)
        loss = criterion(output, targets)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        loss, output, targets = accelerator.gather_for_metrics((loss, output, targets))
        loss = loss.mean()
        total_loss += loss.item()
        stream.set_description(
            f' Training Fold: {fold}, Epoch: {epoch}, Loss: {total_loss / (i + 1):3f}')

        accelerator.log(
            {f'fold_{fold}/train_loss': loss.item(),
             f'fold_{fold}/train_step': (len(train_loader) * epoch) + i})


def valid_fn(
        valid_loader,
        model,
        accelerator,
        fold,
        epoch,
        criterion,
):
    model.eval()
    total_loss = 0
    stream = tqdm(valid_loader, disable=not accelerator.is_local_main_process)
    for i, batch in enumerate(stream):
        targets = batch.pop('targets')
        with torch.no_grad():
            output = model(batch)
        loss = criterion(output, targets)
        loss, output, targets = accelerator.gather_for_metrics((loss, output, targets))
        loss = loss.mean()
        total_loss += loss.item()
        stream.set_description(
            f' Validation Fold: {fold}, Epoch: {epoch}, Loss: {total_loss / (i + 1):3f}')

    accelerator.log({f'fold_{fold}/valid_loss': total_loss / len(valid_loader),
                     f'fold_{fold}/epoch': epoch})
    return total_loss / len(valid_loader)
