import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from utils import load_data


def initialize_parameters(dimensions):
    params = {}
    L = len(dimensions)
    np.random.seed(1)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1])
        params['b' + str(l)] = np.random.randn(dimensions[l], 1)
    return params


def forward_propagation(X, params):
    activations = {'A0': X}
    L = len(params) // 2

    for l in range(1, L + 1):
        Z = params['W' + str(l)].dot(activations['A' + str(l - 1)]) + params['b' + str(l)]
        activations['A' + str(l)] = 1 / (1 + np.exp(-Z))
    return activations


def back_propagation(y, params, activations):
    m = y.shape[1]
    L = len(params) // 2
    dZ = activations['A' + str(L)] - y
    gradients = {}

    for l in reversed(range(1, L + 1)):
        gradients['dW' + str(l)] = 1 / m * np.dot(dZ, activations['A' + str(l - 1)].T)
        gradients['db' + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dZ = np.dot(params['W' + str(l)].T, dZ) * activations['A' + str(l - 1)] * (
                        1 - activations['A' + str(l - 1)])
    return gradients


def update_parameters(gradients, params, learning_rate):
    L = len(params) // 2
    for l in range(1, L + 1):
        params['W' + str(l)] = params['W' + str(l)] - learning_rate * gradients['dW' + str(l)]
        params['b' + str(l)] = params['b' + str(l)] - learning_rate * gradients['db' + str(l)]
    return params


def predict(X, params):
    activations = forward_propagation(X, params)
    L = len(params) // 2
    Af = activations['A' + str(L)]
    return Af >= 0.5


def deep_neural_network(X_train, y_train, X_test, y_test, hidden_layers=(16, 16, 16), learning_rate=0.001, n_iter=3000):
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    params = initialize_parameters(dimensions)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    L = len(params) // 2

    for i in tqdm(range(n_iter), desc="Training Deep NN"):
        activations_train = forward_propagation(X_train, params)
        gradients = back_propagation(y_train, params, activations_train)
        params = update_parameters(gradients, params, learning_rate)

        Af_train = activations_train['A' + str(L)]
        train_loss.append(log_loss(y_train.flatten(), Af_train.flatten()))
        y_pred_train = predict(X_train, params)
        train_acc.append(accuracy_score(y_train.flatten(), y_pred_train.flatten()))

        activations_test = forward_propagation(X_test, params)
        Af_test = activations_test['A' + str(L)]
        test_loss.append(log_loss(y_test.flatten(), Af_test.flatten()))
        y_pred_test = predict(X_test, params)
        test_acc.append(accuracy_score(y_test.flatten(), y_pred_test.flatten()))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.legend()
    plt.show()

    return params


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    X_train_flatten = X_train.reshape(X_train.shape[0], -1).T / 255.0
    X_test_flatten = X_test.reshape(X_test.shape[0], -1).T / 255.0
    y_train_reshaped = y_train.T
    y_test_reshaped = y_test.T

    final_params = deep_neural_network(
        X_train_flatten, y_train_reshaped, X_test_flatten, y_test_reshaped,
        hidden_layers=(32, 32, 16), learning_rate=0.01, n_iter=8000
    )

    final_accuracy = accuracy_score(y_test, predict(X_test_flatten, final_params).T)
    print(f"\nFinal accuracy on the test set: {final_accuracy:.2%}")