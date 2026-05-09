from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

np.random.seed(42)

class DecisionTree:
    def __init__(self, max_depth, min_samples_leaf):
        self.max_depth = max_depth;
        self.min_samples_leaf = min_samples_leaf;
        self.root=None;


    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)


    def build_tree(self, X, y, depth):
        #check for stopping criteria
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_leaf:
            return self.majority_class(y)
        # find best split
        best_feature, best_threshold = self.best_split(X, y)
        if best_feature is None:
            return self.majority_class(y)
        # create child nodes
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_child = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        return (best_feature, best_threshold, left_child, right_child)
            
    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        for feature in range(X.shape[1]):
            thresholds = np.percentile(X[:, feature], np.linspace(5, 95, 20))
            thresholds = np.unique(thresholds)
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                gini_left = self.gini(y[left_indices])
                gini_right = self.gini(y[right_indices])
                gini_split = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / len(y)
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold


    def gini(self, y):
        if len(y) == 0:
            return 0
        y = y.astype(int)  
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    

    def predict(self, X):
        return np.array([self.predict_single(x, self.root) for x in X])
    

    def predict_single(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left_child, right_child = node
        if x[feature] < threshold:
            return self.predict_single(x, left_child)
        else:
            return self.predict_single(x, right_child)
        

    def majority_class(self, y):
        y = y.astype(int)
        return np.argmax(np.bincount(y)) 



mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy()
y = mnist.target.to_numpy().astype(int)
X = X / 255.0
y = (y == 8).astype(int) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_8 = X_train[y_train == 1]
X_not8 = X_train[y_train == 0]

idx = np.random.choice(len(X_not8), size=len(X_8), replace=False)

#X_train = np.vstack((X_8, X_not8[idx]))
#y_train = np.hstack((np.ones(len(X_8)), np.zeros(len(X_8))))
#y_train = y_train.astype(int)

X_train= np.array([hog(X_train[i].reshape(28, 28), pixels_per_cell=(4, 4), cells_per_block=(2, 2)) for i in range(len(X_train))])
X_test= np.array([hog(X_test[i].reshape(28, 28), pixels_per_cell=(4, 4), cells_per_block=(2, 2)) for i in range(len(X_test))])


model = DecisionTree(max_depth=7, min_samples_leaf=30)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy, Precision, Recall, F1-score, Confusion Matrix
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

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)
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
plt.title('Confusion Matrix Heatmap')
plt.show()