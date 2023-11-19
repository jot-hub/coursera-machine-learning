import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_data_multi
from public_tests import compute_gradient_test
import copy
import math

from sklearn.linear_model import SGDRegressor

# load the dataset
x_train, y_train = load_data()
x_train = x_train.reshape(-1,1)

sgdr = SGDRegressor(max_iter=1500)
sgdr.fit(x_train, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b = sgdr.intercept_
w = sgdr.coef_
print(f"model parameters:                   w: {w}, b:{b}")

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted.put(i, w * (x_train.item(i)) + b)

predict1 = 3.5 * w + b
print(f"predict1 shape = {predict1.shape}")
print('For population = 35,000, we predict a profit of $%.2f' % (predict1.item(0)*10000))

predict2 = 7.0 * w + b
print(f"predict2 shape = {predict2.shape}")
print('For population = 70,000, we predict a profit of $%.2f' % (predict2.item(0)*10000))