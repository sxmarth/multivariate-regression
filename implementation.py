import numpy as np
import matplotlib.pyplot as plt

#Step 1: Initialization of parameters
def initialize_parameters(lenw):
    w = np.zeros((1, lenw))
    b = 0
    return w, b

#Step 2: Forward Propagation
def forward_prop(X, w, b)
    z = np.dot(w,X) + b
    return z

#Step 3: