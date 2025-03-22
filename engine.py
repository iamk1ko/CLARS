import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


def train_one_epoch_classification(train_loader,
                                   model,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   epoch,
                                   logger,
                                   config,
                                   scaler=None):
    if config.amp and scaler is None:
        scaler = torch.cuda.amp.GradScaler()

    stime = time.time()
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        img, label, img_name = data
        img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)

        optimizer.zero_grad()

        if config.amp:
            with autocast():
                out = model(img)
                if type(out) is dict:
                    out = out['out']
                loss = criterion(out, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(img)
            loss = criterion(out, label)
            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, batch_loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

        if config.scheduler_batch_step:
            scheduler.step()

    if not config.scheduler_batch_step:
        scheduler.step()

    mean_loss = np.mean(loss_list)
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, mean_loss: {mean_loss:.4f}, time(s): {etime - stime:.2f}'
    print(log_info)
    logger.info(log_info)

    return mean_loss


def val_one_epoch_classification(val_loader, model, criterion, epoch, logger):
    model.eval()

    loss_list = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(val_loader):
            img, label, img_name = data
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)

            out = model(img)
            if type(out) is dict:
                out = out['out']
            loss = criterion(out, label)

            loss_list.append(loss.item())

            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    log_info = (f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, accuracy: {accuracy:.4f}, '
                f'precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')
    print(log_info)
    logger.info(log_info)

    return np.mean(loss_list)


def pred_one_epoch_classification(test_loader, model, logger=None):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            img, label, img_name = data
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True).long()

            out = model(img)
            if type(out) is dict:
                out = out['out']

            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    log_info = (f'test, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, '
                f'f1: {f1:.4f}')
    print(log_info)
    if logger is not None:
        logger.info(log_info)

    return accuracy, precision, recall, f1
