import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load & Prepare Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test  = X_test.reshape(-1, 784)  / 255.0

y_train = (y_train == 8).astype(int)
y_test  = (y_test  == 8).astype(int)


y_train_svm = np.where(y_train == 1, 1, -1)
y_test_svm  = np.where(y_test  == 1, 1, -1)


class linearSVM:
    def __init__(self, C=0.1 , Learning_rate=0.001, n_epochs=50):
        self.C= C
        self.Learing_rate= Learning_rate
        self.n_epochs= n_epochs
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            for idx, sample in enumerate(X):
                margin = y[idx] * (np.dot(sample, self.weights) + self.bias)

                if margin >= 1:
                    self.weights -= self.Learing_rate * (2 / num_samples) * self.weights
                else:
                    self.weights -= self.Learing_rate * ((2 / num_samples) * self.weights - self.C * y[idx] * sample)
                    self.bias += self.Learing_rate * self.C * y[idx]
    
    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.sign(scores)
    
svm = linearSVM(C=0.1, Learning_rate=0.001, n_epochs=50)
svm.fit(X_train, y_train_svm)
y_pred_svm = svm.predict(X_test)

y_pred = np.where(y_pred_svm == 1, 1, 0)

print(f"Accuracy  : {accuracy_score(y_test, y_pred)  * 100:.2f}%")
print(f"Recall    : {recall_score(y_test, y_pred)    * 100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"F1-Score  : {f1_score(y_test, y_pred)        * 100:.2f}%")

# Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["Not 8", "Is 8"]).plot(cmap="Blues")
plt.title("Linear SVM Confusion Matrix")
plt.tight_layout()
plt.show()