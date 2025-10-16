import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from utils import load_data


def initialize_parameters(n0, n1, n2):
    params = {
        'W1': np.random.randn(n1, n0),
        'b1': np.zeros((n1, 1)),
        'W2': np.random.randn(n2, n1),
        'b2': np.zeros((n2, 1))
    }
    return params


def forward_propagation(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        'A1': A1,
        'A2': A2
    }
    return activations


def back_propagation(X, y, params, activations):
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = params['W2']
    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }
    return gradients


def update_parameters(gradients, params, learning_rate):
    W1 = params['W1'] - learning_rate * gradients['dW1']
    b1 = params['b1'] - learning_rate * gradients['db1']
    W2 = params['W2'] - learning_rate * gradients['dW2']
    b2 = params['b2'] - learning_rate * gradients['db2']

    params = {
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2
    }
    return params


def predict(X, params):
    activations = forward_propagation(X, params)
    A2 = activations['A2']
    return A2 >= 0.5


def neural_network(X_train, y_train, X_test, y_test, n1=32, learning_rate=0.01, n_iter=10000):
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    np.random.seed(0)
    params = initialize_parameters(n0, n1, n2)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in tqdm(range(n_iter), desc="Training Shallow NN"):
        activations = forward_propagation(X_train, params)

        gradients = back_propagation(X_train, y_train, params, activations)
        params = update_parameters(gradients, params, learning_rate)

        train_loss.append(log_loss(y_train.flatten(), activations['A2'].flatten()))
        y_pred_train = predict(X_train, params)
        train_acc.append(accuracy_score(y_train.flatten(), y_pred_train.flatten()))

        activations_test = forward_propagation(X_test, params)
        test_loss.append(log_loss(y_test.flatten(), activations_test['A2'].flatten()))
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

    X_train_flatten = X_train.reshape(X_train.shape[0], -1).T / X_train.max()
    X_test_flatten = X_test.reshape(X_test.shape[0], -1).T / X_train.max()
    y_train_reshaped = y_train.T
    y_test_reshaped = y_test.T

    m_train = 300
    m_test = 80

    X_train_subset = X_train_flatten[:, :m_train]
    X_test_subset = X_test_flatten[:, :m_test]

    y_train_subset = y_train_reshaped[:, :m_train]
    y_test_subset = y_test_reshaped[:, :m_test]

    final_params = neural_network(X_train_subset, y_train_subset, X_test_subset, y_test_subset, n1=32,
                                  learning_rate=0.01, n_iter=10000)

    y_test_correct_shape = y_test[:m_test]

    final_accuracy = accuracy_score(y_test_correct_shape, predict(X_test_subset, final_params).T)
    print(f"\nFinal accuracy on the test set: {final_accuracy:.2%}")