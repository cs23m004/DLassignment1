import numpy as np

# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rYAkEjPQ8RG0tP1AmUf-MjtW9Kq_HRy7
"""
#------------------------------------------------------------------ Helping Functions------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist

def sigmoid(x):
  var = 1/(1+np.exp(-x))
  return var

def sigmoid_der(x):
  var = sigmoid(x)
  var = var * (1-var)
  return var

def relu(x):
  var = np.maximum(0, x)
  return var

def relu_der(x):
  x[x>0] = 1
  x[x<=0] = 0
  return x

def tanh(x):
  varp = np.exp(x)
  varn = np.exp(-x)
  res = (varp - varn) / (varp + varn)
  return res

def tanh_der(x):
  der = 1 - tanh(x) ** 2
  return der

def identity(x):
  return x

def identity_der(x):
  der = np.ones(x.shape)
  return der

def get_activation(act):
  if act == 'sigmoid':
    return sigmoid, sigmoid_der
  if act == 'ReLU':
    return relu, relu_der
  if act== 'tanh':
    return tanh, tanh_der
  if act == 'identity':
    return identity, identity_der
  raise Exception('Error : Wrong activation function called')

def get_loss_func(loss_func):
  if loss_func == 'cross_entropy':
    return cross_entropy, cross_entropy_der
  if loss_func == 'mean_squared_error':
    return mean_squared_error, mean_squared_error_der
  raise Exception('Error : Wrong loss function called')

def softmax(x):
  expo = np.exp(x)
  temp = np.sum(expo, axis=1, keepdims=True)
  res = expo/temp
  return res

def cross_entropy(y, yhat):
  epsilon = 1e-30
  yhat = yhat + epsilon
  res = y * np.log(yhat)
  losses = -np.sum(res, axis=1)
  meanloss= np.mean(losses)
  return meanloss

def cross_entropy_der(y, yhat):
  res = -(y-yhat)
  return res

def mean_squared_error(y, yhat):
  temp = (y-yhat)**2
  temp = np.sum(temp,axis=1)
  res = np.mean(temp)
  return res

def mean_squared_error_der(y, yhat):
  temp = ((y-yhat) * yhat)
  res = np.sum(temp, axis=1, keepdims=True)
  res = yhat * (res - (y-yhat))
  return res

def accuracy(y, yhat):
  predicted_labels = np.argmax(yhat, axis=1)
  true_labels = np.argmax(y, axis=1)
  correct_predictions = np.sum(predicted_labels == true_labels)
  acc = correct_predictions / len(y)
  return acc

def get_dataset(dataset):
    datasets = {
        'fashion_mnist': (fashion_mnist.load_data(),
                          {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
                           4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker',
                           8: 'Bag', 9: 'Ankle Boot'}),

        'mnist': (mnist.load_data(),
                  {i: str(i) for i in range(10)})
    }

    if dataset not in datasets:
        raise ValueError('Error : Wrong Dataset Called')

    (xfull, yfull), (xtest, ytest), class_labels = *datasets[dataset][0], datasets[dataset][1]

    np.random.seed(2)
    test_size = 0.1
    ltrain = int(len(xfull) * test_size)

    idxs = np.random.permutation(len(xfull))
    xval, yval = xfull[idxs[:ltrain]], yfull[idxs[:ltrain]]
    xtrain, ytrain = xfull[idxs[ltrain:]], yfull[idxs[ltrain:]]

    return (xtrain, ytrain), (xval, yval), (xtest, ytest), class_labels


def scale_dataset(xtrain, ytrain, xval, yval, xtest, ytest, scaling):
    def min_max_scaling(x):
        return x.reshape((x.shape[0], -1)) / 255.0

    def standard_scaling(x, mu=None, sigma=None):
        x = x.reshape((x.shape[0], -1))
        if mu is None or sigma is None:
            mu, sigma = x.mean(axis=0), x.std(axis=0)
        return (x - mu) / sigma, mu, sigma

    def one_hot_encode(y):
        return np.array(pd.get_dummies(y))

    if scaling == 'min_max':
        xtrain_inp, xval_inp, xtest_inp = map(min_max_scaling, [xtrain, xval, xtest])

    elif scaling == 'standard':
        xtrain_inp, mu, sigma = standard_scaling(xtrain)
        xval_inp, _, _ = standard_scaling(xval, mu, sigma)
        xtest_inp, _, _ = standard_scaling(xtest, mu, sigma)

    else:
        raise ValueError('Error : Wrong Data Scaling name called')

    ytrain_inp, yval_inp, ytest_inp = map(one_hot_encode, [ytrain, yval, ytest])

    return (xtrain_inp, ytrain_inp), (xval_inp, yval_inp), (xtest_inp, ytest_inp)

#-----------------------------------------------------------------------------------Neural Network (Forward, backward, inits etc)------------------------------------------------------------------------------------


def init_params(input_num, hidden_size, output_num, init_type, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    Layer = [input_num] + hidden_size + [output_num]
    
    def initialize_weights(fan_in, fan_out, method):
        if method == 'random':
            return np.random.randn(fan_in, fan_out) * 0.01
        elif method == 'Xavier':
            val1 = np.random.randn(fan_in, fan_out)
            val2 = np.sqrt(2 / (fan_in + fan_out))
            init = val1 * val2
            return init
        else:
            raise ValueError("Error : invalid init type")

    params = {
        f"W{i}": initialize_weights(Layer[i-1], Layer[i], init_type) for i in range(1, len(Layer))
    }
    params.update({f"B{i}": np.zeros((1, Layer[i])) for i in range(1, len(Layer))})

    return params

def forward(inp, params, activation):
    L = len(params) // 2
    layer_cal = {'H0': inp}

    for i in range(1, L):
        A = layer_cal[f'H{i-1}'] @ params[f'W{i}'] + params[f'B{i}']
        layer_cal[f'A{i}'], layer_cal[f'H{i}'] = A, activation(A)

    A_final = layer_cal[f'H{L-1}'] @ params[f'W{L}'] + params[f'B{L}']
    layer_cal[f'A{L}'], layer_cal[f'H{L}'] = A_final, softmax(A_final)

    return layer_cal[f'H{L}'], layer_cal


def eval_params(x, y, params, config):
    act, _ = get_activation(config['activation'])
    loss_func, _ = get_loss_func(config['loss_func'])
    
    yhat, _ = forward(x, params, act)
    reg_term = 0.5 * config['WD'] * sum(np.linalg.norm(w) for w in params.values())
    
    return loss_func(y, yhat) + reg_term, accuracy(y, yhat)


def backward(y, params, yhat, layer_cal, act_der, loss_func_der, WD):
    L, m = len(params) // 2, y.shape[0]
    del_params, del_ak = {}, loss_func_der(y, yhat)
    
    for k in range(L, 0, -1):
        del_params[f'W{k}'] = (layer_cal[f'H{k-1}'].T @ del_ak + WD * params[f'W{k}']) / m
        del_params[f'B{k}'] = np.sum(del_ak, axis=0, keepdims=True) / m
        if k > 1:
            del_ak = (del_ak @ params[f'W{k}'].T) * act_der(layer_cal[f'A{k-1}'])

    return del_params

def predict(inp, params, config):
    yhat, _ = forward(inp, params, get_activation(config['activation'])[0])
    return np.argmax(yhat, axis=1)