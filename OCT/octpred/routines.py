import time
import copy

import os
from absl import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils.metrics import Metrics
from .utils.vis import visualize_model

TRAIN='train'
VAL='val'
TEST='test'
use_gpu = torch.cuda.is_available()

def maybe_restore(model, weights_dir):
    try:
        model.load_state_dict(torch.load(weights_dir))
    except FileNotFoundError:
        logging.debug("Weights file at {} not found".format(weights_dir))

def save_model(model, weights_dir):
    torch.save(model.state_dict(), weights_dir)

def train_step(model, inputs, labels, optimizer, criterion):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return outputs, loss

def deploy_model(model, ds, mode="val", save_path=None):
    visualize_model(model, ds, mode, save_dir=save_path)

def eval_model(model, criterion, ds, mode="val"):

    since = time.time()
    met = Metrics(initial_value_dict={"loss": 0., "acc": 0.})
    avg_loss = 0
    avg_acc = 0
    dataloader = ds.get_dataloaders()[mode]
    dataset_size = ds.get_dataset_sizes()[mode]
    test_batches = len(dataloader)
    logging.info("Evaluating model")
    logging.info('-' * 10)

    for i, data in tqdm(enumerate(dataloader)):
        if i % 100 == 0:
            logging.info("\rTest batch {}/{}".format(i, test_batches))
        model.train(False)
        model.eval()
        inputs, labels = data
        with torch.no_grad():
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        met.update({"loss": loss.data, "acc": torch.sum(preds == labels.data)})
        # loss_test += loss.data  # [0] 08/15 JC
        # acc_test += torch.sum(preds == labels.data)
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = met.get_metrics()["loss"] / dataset_size  #dataset_sizes[sel]
    avg_acc = float(met.get_metrics()["acc"]) / dataset_size # dataset_sizes[sel]

    elapsed_time = time.time() - since

    logging.info("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    logging.info("Avg loss (test): {:.4f}".format(avg_loss))
    logging.info("Avg acc (test): {}/{}: {:.4f}".format(met.get_metrics()["acc"], ds.get_dataset_sizes()[mode], avg_acc))
    logging.info('-' * 10)

    return met

def train_model(model, ds, criterion, optimizer, scheduler, num_epochs=1, debug=False, val=True, save_dir=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_train_met = Metrics(initial_value_dict={"loss": 0., "acc": 0.})
    avg_val_met = Metrics(initial_value_dict={"loss": 0., "acc": 0.})

    met_train = Metrics(initial_value_dict={"loss": 0., "acc": 0.})
    met_val = Metrics(initial_value_dict={"loss": 0., "acc": 0.})

    dataloaders = ds.get_dataloaders()
    dataset_sizes = ds.get_dataset_sizes()
    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])
    if debug:
        num_epochs = 1
        max_iter = 4
    else:
        max_iter = 1 + train_batches

    for epoch in range(num_epochs):
        logging.info("Epoch {}/{}".format(epoch, num_epochs))
        logging.info('-' * 10)
        # reset epoch metrics
        met_train.reset()
        met_val.reset()
        if save_dir:
            save_model(model, os.path.join(save_dir, "epoch-{:04d}.pt".format(epoch)))

        model.train(True)

        for i, data in tqdm(enumerate(dataloaders[TRAIN])):
            if i % 100 == 0:
                if i > 0:
                    logging.info("\rTraining batch {}/{}: Loss = {:.4f}".format(i, train_batches, loss.data))

            if i > max_iter:
                break

            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs, loss = train_step(model, inputs, labels, optimizer, criterion)
            _, preds = torch.max(outputs.data, 1)
            met_train.update({"loss": loss.data, "acc": torch.sum(preds == labels.data)})

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        epoch_metrics = met_train.get_metrics()
        avg_loss = epoch_metrics["loss"] / dataset_sizes[TRAIN]
        avg_acc = float(epoch_metrics["acc"]) / dataset_sizes[TRAIN]
        avg_train_met.update({"loss": avg_loss, "acc": avg_acc})

        model.train(False)
        model.eval()

        for i, data in tqdm(enumerate(dataloaders[VAL])):
            if i % 100 == 0:
                logging.info("\rValidation batch {}/{}".format(i, val_batches))

            optimizer.zero_grad()
            inputs, labels = data
            with torch.no_grad():
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            met_val.update({"loss": loss.data, "acc": torch.sum(preds == labels.data)})

        avg_loss_val = met_val.get_metrics()["loss"] / dataset_sizes[VAL]
        avg_acc_val = float(met_val.get_metrics()["acc"]) / dataset_sizes[VAL]
        avg_val_met.update({"loss": avg_loss_val, "acc": avg_acc_val})

        logging.info("Epoch {} result: ".format(epoch))
        logging.info("Avg loss (train): {:.4f}".format(avg_loss))
        logging.info("Avg acc (train): {}/{}: {:.4f}".format(met_train.get_metrics()["acc"], dataset_sizes[TRAIN], avg_acc))
        logging.info("Avg loss (val): {:.4f}".format(avg_loss_val))
        logging.info("Avg acc (val): {}/{}: {:.4f}".format(met_val.get_metrics()["acc"], dataset_sizes[VAL], avg_acc_val))
        logging.info('-' * 10)

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed_time = time.time() - since

    logging.info("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    logging.info("Best acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    if save_dir:
        save_model(model, os.path.join(save_dir, "session-best.pt"))

    return model