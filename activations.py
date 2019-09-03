import numpy as np

def g(x):
    return 1 / (1 + np.exp(-x))
 
def gp(x):
    A = g(x)
    return A * (1-A)
 
def gpp(x):
    A = g(x)
    return A * (1-A) * (1-(2*A))
   
# activation functions and their derivatives
activations = {
    "sigmoid": (g, gp, gpp),
    "sine": (np.sin, np.cos, lambda x: -np.sin(x))
}
