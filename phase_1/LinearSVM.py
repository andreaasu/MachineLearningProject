import numpy as np
from tensorflow.keras.datasets import mnist
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

tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

cm = np.array([[tn, fp],
               [fn, tp]])

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

# Confusion Matrix
plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest')
plt.colorbar()
plt.xticks([0,1], ['Predicted Not 8', 'Predicted 8'])
plt.yticks([0,1], ['Actual Not 8', 'Actual 8'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Linear SVM Confusion Matrix')
plt.tight_layout()
plt.show()