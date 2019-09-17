import torch

def gp(x):
    A = torch.sigmoid(x)
    return A * (1-A)
 
def gpp(x):
    A = torch.sigmoid(x)
    return A * (1-A) * (1-(2*A))
   
# activation functions and their derivatives
activations = {
    "sigmoid": (torch.sigmoid, gp, gpp),
    "sine": (torch.sin, torch.cos, lambda x: -torch.sin(x))
}

