import numpy as np
from data_loader import load_mnist
from utils import print_metrics

def sigmoid(z):
    # squashes any real number into a probability between 0 and 1
    return 1 / (1 + np.exp(-z))

def calculate_gradient(theta, X, y):
    # average gradient of the binary cross-entropy loss
    m = y.size
    return X.T @ (sigmoid(X @ theta) - y) / m

def compute_loss(y, y_hat):
    # binary cross-entropy loss with clipping for numerical stability
    m = y.size
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def predict_prob(X, theta):
    # add bias column then compute probability
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

class LogisticRegression:
    def __init__(self, alpha=0.1, num_iter=100, tolerance=1e-7):
        self.alpha = alpha
        self.num_iter = num_iter
        self.tolerance = tolerance
        self.theta = None
        self.losses = []
        self.val_losses = []
        self.actual_iter = 0

    def fit(self, X, y, X_val=None, y_val=None):
        # add bias column of ones
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.zeros(X_b.shape[1])

        self.losses = []
        self.val_losses = []

        for i in range(self.num_iter):
            y_hat = sigmoid(X_b @ self.theta)
            loss = compute_loss(y, y_hat)
            self.losses.append(loss)
            self.actual_iter = i + 1

            # track validation loss to spot overfitting early
            if X_val is not None and y_val is not None:
                X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
                y_val_hat = sigmoid(X_val_b @ self.theta)
                val_loss = compute_loss(y_val, y_val_hat)
                self.val_losses.append(val_loss)

            grad = calculate_gradient(self.theta, X_b, y)
            self.theta -= self.alpha * grad

            # stop when the gradient is nearly zero (converged)
            if np.linalg.norm(grad) < self.tolerance:
                break
        return self.theta

    def predict(self, X, threshold=0.5):
        # probability >= threshold means positive class
        return (predict_prob(X, self.theta) >= threshold).astype(int)

X_train, X_val, X_test, y_train, y_val, y_test = load_mnist()

model = LogisticRegression(alpha=1.3, num_iter=2000)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
print(f"iterations: {model.actual_iter}")
if model.losses:
    print(f"final train loss: {model.losses[-1]:.4f}")
if model.val_losses:
    print(f"final val loss: {model.val_losses[-1]:.4f}")

y_pred_train = model.predict(X_train, threshold=0.4)
y_pred_val = model.predict(X_val, threshold=0.4)
y_pred_test = model.predict(X_test, threshold=0.4)
print_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, model)
