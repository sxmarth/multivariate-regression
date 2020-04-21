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

#Step 6: Define Linear Regression Model
def linear_reg_model(X_train, y_train, X_val, y_val, learning_rate, epochs):
    lenw = X_train.shape[0]
    w,b = initialize_parameters(lenw)

    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]

    for i in range(1, epochs+1):
        z_train = forward_prop(X_train, w, b)
        cost_train = cost_function(z_train,y_train)
        dw,db = back_prop(X_train, y_train, z_train)
        w,b = gradient_descent_update(w, b, dw, db, learning_rate)

        #Storing training cost in a list to plot
        if i%10==0:
            costs_train.append(cost_train)
        MAE_train = (1/m_train) np.sum(np.abs(z_train-y_train))

        #Finding cost function value
        z_val = forward_prop(X_val,w,b)
        cost_val = cost_function(z_val, y_val)
        MAE_train = (1/m_val)*np.sum(np.abs(z_train-y_train))

        #Printing Values
        print('Epochs' +str(i)+'/'+str(epochs)+': ')
        print('Tech cost'+str(cost_train)+'|'+'Validation cost' +str(cost_val))
        print('Tech cost'+str(MAE_train)+'|'+'Validation MAE' +str(MAE_val))

    plt.plot(costs_train)
    plt.xlabel('Iterations(per tens)')
    plt.ylabel('Training Cost')
    plt.title('Learning rate'+str(learning_rate))
    plt.show()