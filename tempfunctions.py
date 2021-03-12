import torch
import torch.nn as nn

def is_cuda(device):
    """ Returns whether cuda will be used in the current running program."""
    return device and device == torch.device('cuda')

def set_cuda(device_type):
    if device_type == 'cuda':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    elif device_type == 'cpu':
        return torch.device('cpu')
    else:
        print('Unexpected device type: ' + device_type)
        return torch.device(device_type)

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
