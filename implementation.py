import numpy as np
import matplotlib.pyplot as plt

#Step 1: Initialization of parameters
def initialize_parameters(lenw):
    w = np.zeros((1, lenw))
    b = 0
    return w, b

#Step 2: Forward Propagation
def forward_prop(X, w, b):
    z = np.dot(w,X) + b
    return z

#Step 3: Cost/Loss Function
def cost_function(z, y):
    J = (1/(2*m))*np.sum(np.square(z-y))
    return J

#Step 4: Backpropagation
def back_prop(X, y, z):
    m = y.shape[1]
    dz = (1/m)*(z-y)
    dw = np.dot(dz,X.T)
    db = np.sum(dz)
    return dw, db

#Step 5: Gradient Descent Update
def gradient_descent_update(w, b, dw, db, learning_rate):
    w = w - learning_rate*dw
    b = b - learning_rate*db
    return w, b

