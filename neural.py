from collections import OrderedDict
from copy import deepcopy
from tempfunctions import is_cuda, set_cuda
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset


class Neural(nn.Module):
    """ Generic neural network class for simple neural network construction with nn.Sequential"""

    def __init__(self, layers=None, optimizer=None, loss_fn=nn.MSELoss(), device=torch.device('cpu')):
        super(Neural, self).__init__()
        self.device = device
        self.loss = None
        self.loaded_dict = None
        self.set_network(layers)
        self.set_loss_function(loss_fn)
        self.set_optimizer(optimizer)

    def set_loss_function(self, loss_function):
        self.loss_fn = loss_function

    def set_optimizer(self, optimizer):
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            if optimizer == 'adam':
                self.optimizer = optim.Adam(self.parameters())
            elif optimizer == 'sgd':
                self.optimizer = optim.SGD(self.parameters())
            else:
                print('{} is unknown optimizer option'.format(optimizer))
        else:
            print('Likely no optimizer given and no network found')
            return

    def set_optimizer_lambda(self, optimizer):
        """" Sets optimizer using lambda function as optimizer input."""
        if optimizer:
            self.optimizer = optimizer(self.parameters())
        elif not optimizer and self.network:
            self.optimizer = optim.Adam(self.parameters())
        else:
            print('Likely no optimizer given and no network found')

    def set_network(self, network):
        if network is None:
            print("Model has no layers")
            return
        if isinstance(network, nn.Sequential):
            self.network = network
        elif isinstance(network, OrderedDict):
            self.network = nn.Sequential(network)
        else:
            print("Unknown object, can't make into model.")
        # self.network = network if isinstance(network, nn.Sequential) else nn.Sequential(network)

    def forward(self, x, single_entry=False):
        x = torch.unsqueeze(x, 0) if single_entry else x
        x = x.to(self.device) if is_cuda(self.device) else x
        return self.network(x) if self.network else x

    def print_observation(self, input_dataset, index):
        """
        Takes dataset and index inside dataset and prints
        the input, value found from model, and expected value.
        """
        self.eval()
        input_data, correct_data = input_dataset[index]
        print('Input: ', input_data)
        print('Calculated answer: ', self.forward(input_data, single_entry=True))
        print('Correct: ', correct_data)

    def fit(self, data, epochs=1, batch_size=1, print_freq=1):
        """ Convert data to data loader and train."""
        if isinstance(data, tuple):
            data = TensorDataset(data[0], data[1])
        if isinstance(data, TensorDataset):
            data = DataLoader(data, batch_size=batch_size)
        # self.train_with_loader(data, epochs=epochs)
        print_epoch = abs(print_freq if isinstance(print_freq, int) and print_freq > -1 else (print_freq) * epochs)
        for epoch in range(epochs):
            if epoch % print_epoch == 0 and epoch != 0:
                print("Epoch {}".format(epoch))
            self.train()
            for train_in, train_out in data:
                self.compute_loss(train_in, train_out, is_guess=False, training=True)
            self.eval()

    def train_with_loader(self, data, validating_data=None, scheduler=None, epochs=1):
        """
        Trains network using dataloaders with optional validation dataloader.
        """
        print('Training...')
        for epoch in range(epochs):
            self.train()
            for train_in, train_out in data:
                self.compute_loss(train_in, train_out, is_guess=False, training=True)
            self.eval()
            if validating_data:
                with torch.no_grad():
                    valid_loss = self.compute_loss_loader(validating_data).item()
                    print('Average validation error at step ',epoch+1,': ', valid_loss)
                if scheduler and valid_loss:
                    scheduler.step()

    def compute_loss_loader(self, test_data):
        with torch.no_grad():
            total_loss = 0
            self.eval()
            total_loss = sum([self.compute_loss(test_in, test_out, is_guess=False, training=False)
                for test_in, test_out in test_data])
            # for test_in, test_out in test_data:
            #     total_loss += self.compute_loss(test_in, test_out, is_guess=False, training=False)
            return total_loss/len(test_data) if test_data else None

    def compute_loss(self, input_data, correct, is_guess=False, training=False):
        input_data = input_data.to(self.device) if is_cuda(self.device) else input_data
        correct = correct.to(self.device) if is_cuda(self.device) else correct
        self.loss = self.loss_fn(input_data if is_guess else self.forward(input_data), correct)
        if training:
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        return self.loss

    def save(self, filename):
        """ Saves model state dictionary and optimizer state dictionary."""
        torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, filename)

    def load(self, filename, from_device='cpu', to_device='cpu', load_model_state=True, load_strict=False, load_optimizer=True):
        """ Loads model state dictionary and optimizer state dictionary from file."""
        self.loaded_dict = None
        exists = False
        with open(filename):
            exists = True
        if not exists:
            return
        if from_device != to_device:
            self.loaded_dict = torch.load(filename, map_location=torch.device(to_device))
        else:
            self.loaded_dict = torch.load(filename)
        if load_model_state:
            self.load_model_state(strict=load_strict)
        if load_optimizer:
            self.load_optimizer()

    def load_model_state(self, strict=False):
        """ Loads model state from last loaded dictionary."""
        if not self.loaded_dict:
            print('no model loaded')
            return
        missing, unexpected = self.load_state_dict(self.loaded_dict['model_state_dict'], strict=strict)
        if len(missing) > 0:
            print(missing)
        if len(unexpected) > 0:
            print(unexpected)
        self.eval()
        
    def load_optimizer(self):
        """ Loads optimizer from last loaded dictionary."""
        if not self.loaded_dict:
            print('no model loaded')
            return
        self.optimizer.load_state_dict(self.loaded_dict['optimizer_state_dict'])
        self.eval()

    def prune_model(self, prune_fn, amount, prune_params=None, prune_global=True, make_permanent=False):
        """ Applies pruning function globaly to model."""
        if not prune_fn:
            return
        if prune_global:
            if prune_params:
                prune.global_unstructured(prune_params, pruning_method=prune_fn, amount=amount)
                if make_permanent:
                    self.make_pruning_permanent(prune_params)
            else:
                params = list()
                for param in list(self.network):
                    if hasattr(param,'weight') and hasattr(param,'bias'):
                        params.append((param, 'weight'))
                        params.append((param, 'bias'))
                prune.global_unstructured(params, pruning_method=prune_fn, amount=amount)
                if make_permanent:
                    self.make_pruning_permanent(params)
        elif prune_params:
            print('handles local pruning')
        else:
            print('Needs parameters to prune')

    def make_pruning_permanent(self, prune_params=None):
        """ Makes last prune permanent"""
        if not prune_params:
            params = list()
            for param in list(self.network):
                if hasattr(param,'weight_orig'):
                    prune.remove(param, name='weight')
                if hasattr(param,'bias_orig'):
                    prune.remove(param, name='bias')
        else:
            for param, weight in prune_params:
                prune.remove(param, weight)

    def early_bird_prune(self, train_loader, prune_fn, amount, validate_loader=None, min_dist=0.01, consec_matches=5, max_epochs=100, scheduler=None):
        """ Implements early bird pruning method."""
        last_mask = self.__mask_distance()
        print('last mask: ', last_mask)
        prune_params = list()
        filename = 'early_bird_temp'
        for param in list(self.network):
            if hasattr(param,'weight') and hasattr(param,'bias'):
                prune_params.append((param, 'weight'))
                prune_params.append((param, 'bias'))
        curr_matches = 0
        curr_epoch = 0
        curr_mask = self.__mask_distance()
        orig_loss = self.compute_loss_loader(validate_loader).item()
        while curr_matches < consec_matches and curr_epoch < max_epochs:
            self.train_with_loader(train_loader, validating_data=validate_loader)
            self.save(filename)
            self.prune_model(prune_fn, amount, make_permanent=False, prune_params=prune_params)
            last_mask = curr_mask
            curr_mask = self.__mask_distance()
            print('Current mask is ', curr_mask,
            '; Last mask is ', last_mask, ' are close enough? ',
             curr_mask - last_mask <= min_dist)
            self.make_pruning_permanent(prune_params)
            if curr_mask - last_mask <= min_dist:
                curr_matches += 1
            else:
                curr_matches = 0
            if scheduler and validate_loader:
                valid_loss = self.compute_loss_loader(validate_loader).item()
                print("Epoch ", curr_epoch+1, ": ", valid_loss)
                scheduler.step()
            if curr_matches < consec_matches and valid_loss*0.9 > orig_loss:
                print('valid_loss is ', valid_loss, ' original loss ', orig_loss)
                self.load(filename, load_strict=True)

    def __mask_distance(self):
        """ Caclulates mask distance of pruned model"""
        mask_total = 0
        for name, ten in self.named_buffers():
            if '_mask' in name:
                mask_total += torch.sum(ten).item()
        print('mask value',mask_total)
        return mask_total

if __name__ == '__main__':
    # Example run of neural, makes model that tries to guess the sine of a number(currently in radians)
    # Model 
    from math import sin, pi, radians
    device, dtype, data_size = set_cuda('cpu'), torch.float32, int(1e5)
    model_layers = OrderedDict([
        ('fc1', nn.Linear(1, 100)),
        ('r1', nn.ReLU()),
        ('fc2', nn.Linear(100, 1024)),
        ('r2', nn.ReLU()),
        ('fc3', nn.Linear(1024, 1024)),
        ('r3', nn.ReLU()),
        ('fc4', nn.Linear(1024, 1024)),
        ('r4', nn.ReLU()),
        ('fc5', nn.Linear(1024, 100)),
        ('r5', nn.ReLU()),
        ('fc6', nn.Linear(100, 1))
    ])
    model = Neural(model_layers, 'adam')
    print('sine of 0: {}, 0.25pi: {}, 0.5pi: {}, pi: {}, 1.5pi: {}, 2pi: {}'.format(sin(0), sin(pi*0.25), sin(pi*0.5), sin(pi), sin(pi*1.5), sin(pi*2)))
    print(model.forward(torch.tensor([[0], [pi*0.25], [pi*0.5], [pi], [pi*1.5], [pi*2]], device=device, dtype=dtype), single_entry=True))
    x_vals, y_vals = [[radians(val % 360)] for val in range(data_size)], [[sin(radians(val % 360))] for val in range(data_size)]
    x_vals, y_vals = torch.tensor(x_vals, device=device, dtype=dtype), torch.tensor(y_vals, device=device, dtype=dtype)
    model.fit((x_vals, y_vals), epochs=32, batch_size=2, print_freq=4)
    print(model.forward(torch.tensor([[0], [pi*0.25], [pi*0.5], [pi], [pi*1.5], [pi*2]], device=device, dtype=dtype), single_entry=True))
