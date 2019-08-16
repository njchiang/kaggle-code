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

from .utils.metrics import Metrics

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

def predict_step(model, inputs, labels):
    # might want to use torch.from_numpy()
    with torch.no_grad():
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)    
    return model(inputs)

def train_step(model, inputs, labels, optimizer, criterion):
    optimizer.zero_grad()
    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return outputs, loss

def eval_model(model, criterion, ds, mode="val"):
    
    since = time.time()
    met = Metrics(initial_value_dict={"loss": 0, "acc": 0})
    avg_loss = 0
    avg_acc = 0
    dataloader = ds.get_dataloaders()[mode]
    test_batches = len(dataloader)
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in tqdm(enumerate(dataloader)):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)
        model.train(False)
        model.eval()
        inputs, labels = data
        outputs = predict_step(model, inputs, labels)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        met.update({"loss": loss.data, "acc": torch.sum(preds == labels.data)})
        # loss_test += loss.data  # [0] 08/15 JC
        # acc_test += torch.sum(preds == labels.data)
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    # avg_loss = loss_test / ds.get_dataset_sizes()[mode]  #dataset_sizes[sel]
    # avg_acc = acc_test / ds.get_dataset_sizes()[mode] # dataset_sizes[sel]
    avg_loss = met.get_metrics()["loss"] / ds.get_dataset_sizes()[mode]  #dataset_sizes[sel]
    avg_acc = met.get_metrics()["acc"] / ds.get_dataset_sizes()[mode] # dataset_sizes[sel]

    # assert avg_loss == avg_loss_ds, "losses are not equal"
    # assert avg_acc == avg_acc_ds, "averages are not equal"
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)

def train_model(model, ds, criterion, optimizer, scheduler, num_epochs=1, debug=False, val=True, save_dir=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_train_met = Metrics(initial_value_dict={"loss": 0, "acc": 0})
    avg_val_met = Metrics(initial_value_dict={"loss": 0, "acc": 0})
    
    met_train = Metrics(initial_value_dict={"loss": 0, "acc": 0})
    met_val = Metrics(initial_value_dict={"loss": 0, "acc": 0})
    
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
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        # reset epoch metrics 
        if save_dir:
            save_model(model, os.path.join(save_dir, "epoch-{:04d}.pt".format(epoch)))
        
        met_train.reset()
        met_val.reset()
        
        model.train(True)
        
        for i, data in tqdm(enumerate(dataloaders[TRAIN])):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
            
            if i > max_iter:
                break
            
            inputs, labels = data
            outputs, loss = train_step(model, inputs, labels, optimizer, criterion)
            _, preds = torch.max(outputs.data, 1)
            met_train.update({"loss": loss.data, "acc": torch.sum(preds == labels.data)}) 
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        print()
        avg_loss = met_train.get_metrics()["loss"] / dataset_sizes[TRAIN]
        avg_acc = met_train.get_metrics()["acc"] / dataset_sizes[TRAIN]
        avg_train_met.update({"loss": avg_loss, "acc": avg_acc})

        model.train(False)
        model.eval()
            
        for i, data in tqdm(enumerate(dataloaders[VAL])):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
            
            if i > max_iter:
                break
            
            optimizer.zero_grad()
            inputs, labels = data
            outputs = predict_step(model, inputs, labels)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            met_val.update({"loss": loss.data, "acc": torch.sum(preds == labels.data)})
        
        avg_loss_val = met_val.get_metrics()["loss"] / dataset_sizes[VAL]
        avg_acc_val = met_val.get_metrics()["acc"] / dataset_sizes[VAL]
        avg_val_met.update({"loss": avg_loss_val, "acc": avg_acc_val})
        
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    if save_dir:
        save_model(model, os.path.join(save_dir, "session-best.pt"))

    return model