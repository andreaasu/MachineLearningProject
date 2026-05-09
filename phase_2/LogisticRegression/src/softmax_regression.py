import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist_multiclass
from utils import print_metrics, accuracy

def softmax(z):
    # numerically stable softmax (subtract max before exp)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(y, y_hat, theta=None, lambda_=0.0, reg_type='l2'):
    # average negative log likelihood of the true class
    m = y.shape[0]
    log_likelihood = -np.log(y_hat[range(m), y])
    loss = np.sum(log_likelihood) / m
    if lambda_ > 0 and theta is not None:
        # l2 penalty: shrink weights toward zero (bias excluded)
        if reg_type == 'l2':
            loss += (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
        # l1 penalty: push weights to exactly zero (bias excluded)
        elif reg_type == 'l1':
            loss += (lambda_ / m) * np.sum(np.abs(theta[1:]))
    return loss

def calculate_gradient(X, y, y_hat, theta=None, lambda_=0.0, reg_type='l2'):
    # gradient of cross-entropy loss plus optional regularization
    m = y.size
    y_one_hot = np.eye(y_hat.shape[1])[y]
    grad = X.T @ (y_hat - y_one_hot) / m
    if lambda_ > 0 and theta is not None:
        reg_penalty = np.zeros_like(theta)
        # l2 gradient: add penalty term scaled by lambda/m
        if reg_type == 'l2':
            reg_penalty[1:] = (lambda_ / m) * theta[1:]
        # l1 gradient: add sign of weights scaled by lambda/m
        elif reg_type == 'l1':
            reg_penalty[1:] = (lambda_ / m) * np.sign(theta[1:])
        grad += reg_penalty
    return grad

class SoftmaxRegression:
    def __init__(self, alpha=0.1, num_iter=200, tolerance=1e-4, lambda_=0.0, reg_type='l2'):
        self.alpha = alpha
        self.num_iter = num_iter
        self.tolerance = tolerance
        self.lambda_ = lambda_
        self.reg_type = reg_type
        self.theta = None
        self.losses = []
        self.val_losses = []
        self.actual_iter = 0
        self.num_classes = None

    def fit(self, X, y, X_val=None, y_val=None):
        # prepend bias column of ones
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.num_classes = len(np.unique(y))
        self.theta = np.zeros((X_b.shape[1], self.num_classes))
        self.losses = []
        self.val_losses = []

        for i in range(self.num_iter):
            # forward pass: probabilities for each class
            y_hat = softmax(X_b @ self.theta)
            loss = compute_loss(y, y_hat, self.theta, self.lambda_, self.reg_type)
            self.losses.append(loss)
            self.actual_iter = i + 1

            # track validation loss to detect overfitting
            if X_val is not None and y_val is not None:
                X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
                y_val_hat = softmax(X_val_b @ self.theta)
                val_loss = compute_loss(y_val, y_val_hat, self.theta, self.lambda_, self.reg_type)
                self.val_losses.append(val_loss)

            # backward pass: gradient descent update
            grad = calculate_gradient(X_b, y, y_hat, self.theta, self.lambda_, self.reg_type)
            self.theta -= self.alpha * grad

            # early stop when gradient norm is very small
            if np.linalg.norm(grad) < self.tolerance:
                break
        return self.theta

    def predict(self, X):
        # pick the class with the highest probability
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_hat = softmax(X_b @ self.theta)
        return np.argmax(y_hat, axis=1)

def learning_curve(model_class, X_train, y_train, X_val, y_val,
                   train_sizes=None, model_kwargs=None):
    # train model on increasingly large subsets of data
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    if model_kwargs is None:
        model_kwargs = {}
    train_accs = []
    val_accs = []
    m = X_train.shape[0]
    for size in train_sizes:
        n = int(m * size)
        X_sub = X_train[:n]
        y_sub = y_train[:n]
        model = model_class(**model_kwargs)
        model.fit(X_sub, y_sub, X_val=X_val, y_val=y_val)
        train_accs.append(accuracy(y_sub, model.predict(X_sub)))
        val_accs.append(accuracy(y_val, model.predict(X_val)))
    return np.array(train_sizes) * m, np.array(train_accs), np.array(val_accs)

def regularization_sweep(model_class, X_train, y_train, X_val, y_val,
                         lambdas, model_kwargs=None):
    # test different lambda values and record accuracy for each
    if model_kwargs is None:
        model_kwargs = {}
    train_accs = []
    val_accs = []
    for lam in lambdas:
        kwargs = {**model_kwargs, 'lambda_': lam}
        model = model_class(**kwargs)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        train_accs.append(accuracy(y_train, model.predict(X_train)))
        val_accs.append(accuracy(y_val, model.predict(X_val)))
    return np.array(lambdas), np.array(train_accs), np.array(val_accs)

def plot_learning_curve(train_sizes, train_scores, val_scores, title="Learning Curve"):
    plt.figure()
    plt.plot(train_sizes, train_scores, 'o-', label="Train")
    plt.plot(train_sizes, val_scores, 'o-', label="Validation")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_regularization_sweep(lambdas, train_scores, val_scores, reg_type='l2', title=None):
    if title is None:
        title = f"Regularization Sweep ({reg_type.upper()})"
    plt.figure()
    plt.plot(lambdas, train_scores, 'o-', label="Train")
    plt.plot(lambdas, val_scores, 'o-', label="Validation")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def diagnose_bias_variance(train_acc, val_acc, train_loss=None, val_loss=None):
    # gap = train accuracy minus validation accuracy
    # big gap means overfitting (high variance)
    # low accuracy on both means underfitting (high bias)
    gap = train_acc - val_acc
    print("=== bias-variance diagnosis ===")
    print(f"train accuracy: {train_acc:.4f}")
    print(f"val accuracy:   {val_acc:.4f}")
    print(f"gap:            {gap:.4f}")
    if train_loss is not None and val_loss is not None:
        print(f"train loss:     {train_loss:.4f}")
        print(f"val loss:       {val_loss:.4f}")

    if train_acc < 0.7:
        print("result: high bias (underfitting)")
        print("  -> model is too simple and can't learn the training data")
        print("  -> fix: lower lambda, add more features, or run more iterations")
    elif gap > 0.15:
        print("result: high variance (overfitting)")
        print("  -> model memorizes training data but fails on new data")
        print("  -> fix: raise lambda, reduce features, or get more data")
    elif gap > 0.05:
        print("result: mild overfitting")
        print("  -> small gap exists, slight regularization might help")
    else:
        print("result: good fit")
        print("  -> bias and variance are well balanced")

def cross_validate(X, y, k=5, alpha=0.1, num_iter=200, lambda_=0.0, reg_type='l2'):
    # split data into k folds, train on k-1, validate on 1, repeat
    fold_size = len(X) // k
    accs = []
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val_fold = X[start:end]
        y_val_fold = y[start:end]
        X_train_fold = np.concatenate([X[:start], X[end:]])
        y_train_fold = np.concatenate([y[:start], y[end:]])
        model = SoftmaxRegression(alpha=alpha, num_iter=num_iter, lambda_=lambda_, reg_type=reg_type)
        model.fit(X_train_fold, y_train_fold)
        accs.append(np.mean(model.predict(X_val_fold) == y_val_fold))
    return np.mean(accs), accs

# ── stand-alone training when this file is run directly ──
X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_multiclass()

model = SoftmaxRegression(alpha=1.3, num_iter=2000, lambda_=0.01, reg_type='l2')
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

print(f"iterations: {model.actual_iter}")
if model.losses:
    print(f"final train loss: {model.losses[-1]:.4f}")
if model.val_losses:
    print(f"final val loss: {model.val_losses[-1]:.4f}")

# test different lambda values to see regularization effect
lambdas, reg_train, reg_val = regularization_sweep(
    SoftmaxRegression, X_train, y_train, X_val, y_val,
    lambdas=[0, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    model_kwargs={'alpha': 1.3, 'num_iter': 1000}
)
plot_regularization_sweep(lambdas, reg_train, reg_val)

y_pred_train = model.predict(X_train)
y_pred_val   = model.predict(X_val)
y_pred_test  = model.predict(X_test)

print_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, model)

train_acc = accuracy(y_train, y_pred_train)
val_acc   = accuracy(y_val, y_pred_val)
diagnose_bias_variance(train_acc, val_acc, model.losses[-1], model.val_losses[-1])
