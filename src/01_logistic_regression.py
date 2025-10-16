import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import load_data


def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


def initialize_parameters(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A


def compute_gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)


def update_parameters(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)


def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


def train_logistic_regression(X_train, y_train, X_test, y_test, learning_rate=0.1, n_iter=100):
    W, b = initialize_parameters(X_train)

    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for i in tqdm(range(n_iter), desc="Training Logistic Regression"):
        A = model(X_train, W, b)

        if i % 10 == 0:
            train_loss_history.append(log_loss(A, y_train))
            y_pred_train = predict(X_train, W, b)
            train_acc_history.append(accuracy_score(y_train, y_pred_train))

            A_test = model(X_test, W, b)
            test_loss_history.append(log_loss(A_test, y_test))
            y_pred_test = predict(X_test, W, b)
            test_acc_history.append(accuracy_score(y_test, y_pred_test))

        dW, db = compute_gradients(A, X_train, y_train)
        W, b = update_parameters(dW, db, W, b, learning_rate)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='train loss')
    plt.plot(test_loss_history, label='test loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='train acc')
    plt.plot(test_acc_history, label='test acc')
    plt.legend()
    plt.show()

    return (W, b)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    X_train_flatten = X_train.reshape(X_train.shape[0], -1) / X_train.max()
    X_test_flatten = X_test.reshape(X_test.shape[0], -1) / X_train.max()

    print(f"Shape of flattened X_train: {X_train_flatten.shape}")
    print(f"Shape of flattened X_test: {X_test_flatten.shape}")

    W, b = train_logistic_regression(X_train_flatten, y_train, X_test_flatten, y_test, learning_rate=0.01, n_iter=10000)

    final_accuracy = accuracy_score(y_test, predict(X_test_flatten, W, b))
    print(f"\nFinal accuracy on the test set: {final_accuracy:.2%}")