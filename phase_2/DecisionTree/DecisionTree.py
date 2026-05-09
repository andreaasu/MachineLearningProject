from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

np.random.seed(42)



# Decision Tree Class
class DecisionTree:
    def __init__(self, max_depth, min_samples_leaf):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)

    def build_tree(self, X, y, depth):
        if (depth >= self.max_depth
                or len(np.unique(y)) == 1
                or len(y) < self.min_samples_leaf):
            return self.majority_class(y)

        best_feature, best_threshold = self.best_split(X, y)
        if best_feature is None:
            return self.majority_class(y)

        left_idx  = X[:, best_feature] <  best_threshold
        right_idx = X[:, best_feature] >= best_threshold
        left_child  = self.build_tree(X[left_idx],  y[left_idx],  depth + 1)
        right_child = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return (best_feature, best_threshold, left_child, right_child)

    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        n_sample = max(1, int(np.sqrt(n_features)))
        sampled_features = np.random.choice(n_features, n_sample, replace=False)

        for feature in sampled_features:
        #for feature in range(X.shape[1]):
            thresholds = np.percentile(X[:, feature], np.linspace(5, 95, 20))
            thresholds = np.unique(thresholds)

            for threshold in thresholds:
                left_idx  = X[:, feature] <  threshold
                right_idx = X[:, feature] >= threshold
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gini_split = (
                    len(y[left_idx])  * self.gini(y[left_idx]) +
                    len(y[right_idx]) * self.gini(y[right_idx])
                ) / len(y)

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def gini(self, y):
        y = y.astype(int)
        proportions = np.bincount(y, minlength=10) / len(y)
        return 1 - np.sum(proportions ** 2)

    def predict(self, X):
        return np.array([self.predict_single(x, self.root) for x in X])

    def predict_single(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        if x[feature] < threshold:
            return self.predict_single(x, left)
        else:
            return self.predict_single(x, right)

    def majority_class(self, y):
        return np.argmax(np.bincount(y.astype(int), minlength=10))


# Random Forest
class RandomForest:
    def __init__(self, n_trees=5, max_depth=7, min_samples_leaf=30):
        self.n_trees = n_trees
        self.trees = [
            DecisionTree(max_depth, min_samples_leaf)
            for _ in range(n_trees)
        ]

    def fit(self, X, y):
        for tree in self.trees:
            idx = np.random.choice(len(X), len(X), replace=True)  # bootstrap
            tree.fit(X[idx], y[idx])

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), minlength=10).argmax(),
            axis=0, arr=preds
        )


# Hyperparameter Tuning with K-Fold Cross-Validation
def k_fold_split(X, y, k=3):
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k  

    folds = []
    for i in range(k):
        val_idx   = indices[i * fold_size : (i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size],
                                    indices[(i + 1) * fold_size:]])
        folds.append((train_idx, val_idx))
    return folds


def tune_decision_tree(X_train, y_train, k=3):
    depths = [5, 7, 10]
    leaves = [10, 30, 50]
    folds  = k_fold_split(X_train, y_train, k=k)

    best_acc    = 0
    best_params = None

    print(f"\nHyperparameter Tuning ({k}-Fold Cross-Validation)")
    print("-" * 50)

    for d in depths:
        for l in leaves:
            fold_accs = []
            for train_idx, val_idx in folds:
                model = DecisionTree(d, l)
                model.fit(X_train[train_idx], y_train[train_idx])
                preds = model.predict(X_train[val_idx])
                fold_accs.append(accuracy_manual(y_train[val_idx], preds))

            mean_acc = np.mean(fold_accs)
            print(f"  depth={d:2d}, min_leaf={l:2d} → "
                  f"CV Acc = {mean_acc:.4f}")

            if mean_acc > best_acc:
                best_acc    = mean_acc
                best_params = (d, l)

    print(f"\n  Best params: depth={best_params[0]}, "
          f"min_leaf={best_params[1]}, CV Acc={best_acc:.4f}")
    return best_params


# Overfitting / Underfitting Diagnosis via Learning Curves
def learning_curve(n_trees, max_depth, min_samples_leaf,
                   X_train, y_train, X_test, y_test):
    sizes = [1000, 5000, 10000, 30000]
    train_accs = []
    test_accs  = []

    print("\nLearning Curve")
    print("-" * 50)

    for s in sizes:
        model = RandomForest(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        model.fit(X_train[:s], y_train[:s])

        train_acc = accuracy_manual(y_train[:s], model.predict(X_train[:s]))
        test_acc  = accuracy_manual(y_test,      model.predict(X_test))
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        gap = train_acc - test_acc
        if gap > 0.15:
            diagnosis = "Overfitting"
        elif test_acc < 0.70: 
            diagnosis = "Underfitting"
        else:
            diagnosis = "Good fit"

        print(f"  Size {s:>6} → Train: {train_acc:.3f}, "
              f"Test: {test_acc:.3f}, Gap: {gap:.3f}  {diagnosis}")
    return sizes, train_accs, test_accs



# Evaluation Metrics 
def confusion_matrix_manual(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm


def accuracy_manual(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def precision_recall_f1_manual(y_true, y_pred, num_classes):
    cm = confusion_matrix_manual(y_true, y_pred, num_classes)
    precisions, recalls, f1_scores = [], [], []

    for c in range(num_classes):
        tp = cm[c][c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if precision + recall > 0 else 0)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores), cm

def plot_results(cm, sizes, train_accs, test_accs, num_classes):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Random Forest — Results Summary", fontsize=15, fontweight='bold')

    # Plot 1: Confusion Matrix Heatmap
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title("Confusion Matrix", fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot 2: Learning Curve
    ax = axes[1]
    ax.plot(sizes, train_accs, 'o-', color='steelblue',  label='Train Accuracy', linewidth=2)
    ax.plot(sizes, test_accs,  's-', color='darkorange', label='Test Accuracy',  linewidth=2)
    ax.fill_between(sizes, train_accs, test_accs, alpha=0.1, color='gray', label='Gap')
    ax.set_title("Learning Curve", fontsize=13, fontweight='bold')
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Accuracy")
    ax.set_xscale('log')
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])

    # Plot 3: Per-Class Metrics Bar Chart
    ax = axes[2]
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p  = tp / (tp + fp) if tp + fp > 0 else 0
        r  = tp / (tp + fn) if tp + fn > 0 else 0
        f  = 2 * p * r / (p + r) if p + r > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    x = np.arange(num_classes)
    w = 0.25
    ax.bar(x - w, precisions, w, label='Precision', color='steelblue',  alpha=0.85)
    ax.bar(x,     recalls,    w, label='Recall',    color='darkorange', alpha=0.85)
    ax.bar(x + w, f1s,        w, label='F1',        color='seagreen',   alpha=0.85)
    ax.set_title("Per-Class Metrics", fontsize=13, fontweight='bold')
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved: results.png")
    plt.show()



# Data Loading & HOG Feature Extraction
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy() / 255.0
y = mnist.target.to_numpy().astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def extract_hog(data):
    return np.array([
        hog(x.reshape(28, 28), pixels_per_cell=(4, 4), cells_per_block=(2, 2))
        for x in data
    ])

X_train_hog = extract_hog(X_train)
X_test_hog  = extract_hog(X_test)


# Step 1 — Hyperparameter tuning with k-fold CV 
best_depth, best_leaf = tune_decision_tree(X_train_hog, y_train, k=3)

# Step 2 — Train final Random Forest 
rf = RandomForest(n_trees=5, max_depth=best_depth, min_samples_leaf=best_leaf)
rf.fit(X_train_hog, y_train)
y_pred = rf.predict(X_test_hog)

num_classes = len(np.unique(y))
accuracy= accuracy_manual(y_test, y_pred)
precision, recall, f1, cm = precision_recall_f1_manual(y_test, y_pred, num_classes)

print("\nFinal Results:")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1-score : {f1:.4f}")
print("  Confusion Matrix:")
print(cm)

# Step 3 — Learning curve 
sizes, train_accs, test_accs = learning_curve(
    n_trees=3,
    max_depth=best_depth,
    min_samples_leaf=best_leaf,
    X_train=X_train_hog,
    y_train=y_train,
    X_test=X_test_hog,
    y_test=y_test
)

# Step 4 — Plot results
plot_results(cm, sizes, train_accs, test_accs, num_classes)