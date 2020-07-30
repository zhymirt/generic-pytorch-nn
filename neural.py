import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.utils import prune
import matplotlib.pyplot as plt
import numpy as np

use_cuda = False
device = torch.device('cuda') if torch.cuda.is_available() and use_cuda else torch.device('cpu')
print("Cuda available? ", torch.cuda.is_available())

def is_cuda(device):
    """ Returns whether cuda will be used in the current running program."""
    return device and device == torch.device('cuda')

def num_flat_features(x):
    """ Returns number of flat features for tensor."""
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def make_transfer_model(model, num_outputs=1, freeze_layers=False, Layer=nn.Linear):
    """
    Converts pre-made model to be used for transfer learning, does not change input channels from original size.
    """
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
    last_layer_name = list(model.named_modules())[-1][0]
    num_features = getattr(model, last_layer_name).in_features
    setattr(model, last_layer_name, Layer(num_features, num_outputs))

# def imshow(img):
#     img = img/2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2, 0)))
#     plt.show()

class Lambda(nn.Module):
    """
    Wrapper class from pytorch 'what is torch.nn really' tutorial
    to apply lambda functions in nn.Sequential.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class WrappedDataLoader:
    """  Class for wrapping dataloaders."""
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

class neural(nn.Module):
    """ Generic neural network class for simple neural network construction with nn.Sequential"""
    def __init__(self, layers=None, optimizer=None, loss_fn=nn.MSELoss(), device=torch.device('cpu')):
        super(neural, self).__init__()
        self.device = device
        self.loss = None
        self.loaded_dict = None
        self.set_network(layers)
        self.set_loss_function(loss_fn)
        self.set_optimizer(optimizer)

    def set_loss_function(self, loss_function):
        self.loss_fn = loss_function

    def set_optimizer(self, optimizer):
        if optimizer:
            self.optimizer = optimizer
        elif not optimizer and self.network:
            self.optimizer = optim.Adam(self.parameters())
        else:
            print('Likely no optimizer given and no network found')

    def set_optimizer_lambda(self, optimizer):
        """" Sets optimizer using lambda function as optimizer input."""
        if optimizer:
            self.optimizer = optimizer(self.parameters())
        elif not optimizer and self.network:
            self.optimizer = optim.Adam(self.parameters())
        else:
            print('Likely no optimizer given and no network found')

    def set_network(self, network):
        self.network = nn.Sequential(network) if network else None
        if not network:
            print("Model has no layers")

    def forward(self, x, single_entry=False):
        if single_entry:
            x = torch.unsqueeze(x, 0)
        x = x.to(self.device) if is_cuda(self.device) else x
        if self.network:
            return self.network(x)
        return x

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
            for test_in, test_out in test_data:
                total_loss += self.compute_loss(test_in, test_out, is_guess=False, training=False)
            return total_loss/len(test_data) if test_data else None

    def compute_loss(self, input_data, correct, is_guess=False, training=False):
        input_data = input_data.to(self.device) if is_cuda(self.device) else input_data
        correct = correct.to(self.device) if is_cuda(self.device) else correct
        if is_guess:
            self.loss = self.loss_fn(input_data,correct)
        else:
            self.loss = self.loss_fn(self.forward(input_data),correct)
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
        if(load_model_state):
            self.load_model_state(strict=load_strict)
        if(load_optimizer):
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
