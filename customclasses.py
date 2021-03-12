import torch.nn as nn

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
