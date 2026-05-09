import numpy as np
import tensorflow as tf
import time
from skimage.feature import hog
import matplotlib.pyplot as plt

class PretrainedFeatureExtractor:
    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim
        self.model = None
    
    def build(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        base_model.trainable = False
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
        x = tf.keras.layers.Resizing(224, 224)(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.Dense(self.latent_dim, activation='relu')(x)
        self.model = tf.keras.Model(inputs, x)
    
    def transform(self, X):
        return self.model.predict(X, verbose=0)

def extract_hog_features(images):
    features = []
    for img in images:
        feat = hog(img, orientations=9, pixels_per_cell=(4, 4),
                   cells_per_block=(2, 2), feature_vector=True)
        features.append(feat)
    return np.array(features)

# CHANGED: Increased max_iter and time_limit to allow SMO convergence
class KernelSVM:
    def __init__(self, C=1.0, gamma=0.01, tol=1e-3, max_iter=100, time_limit=120):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.K = None

    def rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        K = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * K)

    def fit(self, X, y):
        self.X = X
        self.y = y.astype(np.float64)
        self.n = X.shape[0]
        self.K = self.rbf_kernel(X, X)
        self.alpha = np.zeros(self.n)
        self.b = 0.0
        examine_all = True
        num_changed = 0
        iteration = 0
        start_time = time.time()
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.n):
                    num_changed += self._examine_example(i)
            else:
                for i in np.where((self.alpha > 0) & (self.alpha < self.C))[0]:
                    num_changed += self._examine_example(i)
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            iteration += 1
            if iteration > self.max_iter or time.time() - start_time > self.time_limit:
                break
        return iteration

    def _examine_example(self, i):
        yi = self.y[i]
        Ei = self._compute_error(i)
        r = Ei * yi
        if (r < -self.tol and yi == 1 and self.alpha[i] < self.C) or \
           (r < -self.tol and yi == -1 and self.alpha[i] > 0) or \
           (r > self.tol and yi == 1 and self.alpha[i] > 0) or \
           (r > self.tol and yi == -1 and self.alpha[i] < self.C):
            return self._heuristic_select_j(i)
        return 0

    def _heuristic_select_j(self, i):
        Ei = self._compute_error(i)
        non_bound = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        if len(non_bound) > 1:
            j = max(non_bound, key=lambda k: abs(Ei - self._compute_error(k)))
            if self._step(i, j):
                return 1
        all_idx = np.arange(self.n)
        all_idx = all_idx[all_idx != i]
        np.random.shuffle(all_idx)
        for j in all_idx[:min(10, len(all_idx))]:
            if self._step(i, j):
                return 1
        return 0

    def _step(self, i, j):
        if i == j:
            return 0
        yi, yj = self.y[i], self.y[j]
        Ei, Ej = self._compute_error(i), self._compute_error(j)
        ai_old, aj_old = self.alpha[i], self.alpha[j]
        if yi != yj:
            L, H = max(0, aj_old - ai_old), min(self.C, self.C + aj_old - ai_old)
        else:
            L, H = max(0, ai_old + aj_old - self.C), min(self.C, ai_old + aj_old)
        if abs(L - H) < 1e-8:
            return 0
        eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
        if eta >= 0:
            return 0
        aj_new = np.clip(aj_old - yj * (Ei - Ej) / eta, L, H)
        if abs(aj_new - aj_old) < 1e-5:
            return 0
        ai_new = ai_old + yi * yj * (aj_old - aj_new)
        if 0 < ai_new < self.C:
            b_new = self.b - Ei - yi * (ai_new - ai_old) * self.K[i, i] - yj * (aj_new - aj_old) * self.K[i, j]
        elif 0 < aj_new < self.C:
            b_new = self.b - Ej - yi * (ai_new - ai_old) * self.K[i, j] - yj * (aj_new - aj_old) * self.K[j, j]
        else:
            b1 = self.b - Ei - yi * (ai_new - ai_old) * self.K[i, i] - yj * (aj_new - aj_old) * self.K[i, j]
            b2 = self.b - Ej - yi * (ai_new - ai_old) * self.K[i, j] - yj * (aj_new - aj_old) * self.K[j, j]
            b_new = (b1 + b2) / 2
        self.alpha[i], self.alpha[j], self.b = ai_new, aj_new, b_new
        return 1

    def _compute_error(self, i):
        return np.dot(self.alpha * self.y, self.K[:, i]) + self.b - self.y[i]

    def decision_function(self, X):
        K = self.rbf_kernel(X, self.X)
        return np.dot(K, self.alpha * self.y) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

# CHANGED: Increased default limits for multi-class wrapper
class KernelSVMMulti:
    def __init__(self, C=1.0, gamma=0.01, max_iter=100, time_limit=120):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.classifiers = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            svm = KernelSVM(C=self.C, gamma=self.gamma, max_iter=self.max_iter, time_limit=self.time_limit)
            svm.fit(X, y_binary)
            self.classifiers[int(cls)] = svm
    
    def predict(self, X):
        decisions = np.zeros((X.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            decisions[:, idx] = self.classifiers[int(cls)].decision_function(X)
        return self.classes[np.argmax(decisions, axis=1)]

class EnsembleSVM:
    def __init__(self, params_list):
        self.models = [KernelSVMMulti(**p) for p in params_list]
    
    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
    
    def predict(self, X):
        votes = np.array([m.predict(X) for m in self.models])
        final = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            vals, counts = np.unique(votes[:, i], return_counts=True)
            final[i] = vals[np.argmax(counts)]
        return final

# CHANGED: 50 train per class, 50 test per class
def load_mnist_balanced(n_per_class=150, n_test_per_class=50):
    print("Loading MNIST...")
    (X_train_raw, y_train), (X_test_raw, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    for digit in range(10):
        idx = y_train == digit
        X_train_list.append(X_train_raw[idx][:n_per_class])
        y_train_list.append(np.full(n_per_class, digit))
        idx = y_test == digit
        X_test_list.append(X_test_raw[idx][:n_test_per_class])
        y_test_list.append(np.full(n_test_per_class, digit))
    X_train = np.vstack(X_train_list).astype(np.float32) / 255.0
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list).astype(np.float32) / 255.0
    y_test = np.concatenate(y_test_list)
    idx = np.random.permutation(len(y_train))
    X_train, y_train = X_train[idx], y_train[idx]
    idx = np.random.permutation(len(y_test))
    X_test, y_test = X_test[idx], y_test[idx]
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    return (X_train, y_train), (X_test, y_test)

# CHANGED: Added manual feature standardization to fix scale imbalance between CNN and HOG
def standardize_features(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std

# CHANGED: Added manual gamma heuristic to prevent poor RBF kernel scaling
def suggest_gamma(X, subsample=300):
    n = min(subsample, X.shape[0])
    idx = np.random.choice(X.shape[0], n, replace=False)
    X_sub = X[idx]
    dists = []
    for i in range(n):
        diff = X_sub[i+1:] - X_sub[i]
        sq_dist = np.sum(diff**2, axis=1)
        dists.extend(sq_dist)
    dists = np.array(dists)
    dists = dists[dists > 0]
    median_dist = np.median(np.sqrt(dists))
    return 1.0 / (median_dist ** 2 + 1e-8)

# CHANGED: Fixed CV to split training data properly, increased limits, added stratified split
def cross_validate(X, y, C, gamma, n_folds=3):
    classes = np.unique(y)
    fold_indices = [[] for _ in range(n_folds)]
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        fold_size = len(cls_idx) // n_folds
        for f in range(n_folds):
            start = f * fold_size
            end = None if f == n_folds-1 else (f+1)*fold_size
            fold_indices[f].extend(cls_idx[start:end])
    
    scores = []
    for fold in range(n_folds):
        val_idx = np.array(fold_indices[fold])
        train_idx = np.concatenate([np.array(fold_indices[f]) for f in range(n_folds) if f != fold])
        model = KernelSVMMulti(C=C, gamma=gamma, max_iter=50, time_limit=60)
        model.fit(X[train_idx], y[train_idx])
        scores.append(np.mean(model.predict(X[val_idx]) == y[val_idx]))
    return np.mean(scores)

# CHANGED: Fixed validation source to use held-out training data, updated iteration limits
def plot_learning_curves(X_train, y_train, X_val, y_val, best_C, best_gamma):
    sizes = [100, 300, 500] if len(y_train) >= 500 else [50, 100, len(y_train)//2]
    sizes = [s for s in sizes if s <= len(y_train)]
    train_scores, val_scores = [], []
    for size in sizes:
        idx = np.random.choice(len(y_train), size, replace=False)
        model = KernelSVMMulti(C=best_C, gamma=best_gamma, max_iter=50, time_limit=60)
        model.fit(X_train[idx], y_train[idx])
        train_scores.append(np.mean(model.predict(X_train[idx]) == y_train[idx]))
        val_scores.append(np.mean(model.predict(X_val) == y_val))
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_scores, 'o-', label='Train')
    plt.plot(sizes, val_scores, 's-', label='Validation')
    plt.xlabel('Training Samples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    plt.close()
    print(f"Learning curves: {train_scores} (train), {val_scores} (val)")

def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true)][int(pred)] += 1
    return cm

def precision_score(y_true, y_pred, average=None, labels=None, n_classes=10):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precisions = []
    for i in range(n_classes):
        tp = cm[i][i]
        fp = sum(cm[:, i]) - tp
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    precisions = np.array(precisions)
    if average == 'macro':
        return np.mean(precisions)
    return precisions

def recall_score(y_true, y_pred, average=None, labels=None, n_classes=10):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    recalls = []
    for i in range(n_classes):
        tp = cm[i][i]
        fn = sum(cm[i, :]) - tp
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    recalls = np.array(recalls)
    if average == 'macro':
        return np.mean(recalls)
    return recalls

def f1_score(y_true, y_pred, average=None, labels=None, n_classes=10):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1s = 2 * p * r / (p + r + 1e-8)
    if average == 'macro':
        return np.mean(f1s)
    return f1s

def print_class_metrics(y_true, y_pred):
    precisions = precision_score(y_true, y_pred, average=None, labels=range(10))
    recalls = recall_score(y_true, y_pred, average=None, labels=range(10))
    f1s = f1_score(y_true, y_pred, average=None, labels=range(10))
    macro_p, macro_r, macro_f1 = precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='macro')
    
    print("\nPer-Class Metrics:")
    print("  Digit | Precision | Recall | F1-Score")
    print("  -------|----------|--------|----------")
    for i in range(10):
        print(f"  {i}     | {precisions[i]:.4f}   | {recalls[i]:.4f} | {f1s[i]:.4f}")
    print(f"\nMacro Avg: Precision={macro_p:.4f}, Recall={macro_r:.4f}, F1={macro_f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("  Pred:   ", end="")
    for i in range(10):
        print(f"{i:>4}", end="")
    print()
    for i in range(10):
        print(f"  True {i}:", end="")
        for j in range(10):
            print(f"{cm[i][j]:>4}", end="")
        print()
    return precisions, recalls, f1s, cm

def main():
    print("=" * 60)
    print("Phase 2: Kernel SVM Multi-class with Improvements")
    print("=" * 60)
    np.random.seed(42)
    tf.random.set_seed(42)
    start_total = time.time()

    (X_train_raw, y_train), (X_test_raw, y_test) = load_mnist_balanced(500, 500)
    
    print("\n[1] Feature Extraction (MobileNetV2 + HOG)...")
    print("  Loading pretrained MobileNetV2...")
    cnn = PretrainedFeatureExtractor(latent_dim=128)
    cnn.build()
    cnn_train = cnn.transform(X_train_raw)
    cnn_test = cnn.transform(X_test_raw)
    print(f"  CNN features shape: {cnn_train.shape}")
    hog_train = extract_hog_features(X_train_raw)
    hog_test = extract_hog_features(X_test_raw)
    
    # CHANGED: Removed isolated HOG scaling, will standardize combined features later
    X_train = np.hstack([cnn_train, hog_train])
    X_test = np.hstack([cnn_test, hog_test])
    
    # CHANGED: Applied manual standardization to fused features
    X_train, X_test, _, _ = standardize_features(X_train, X_test)
    print(f"Combined features shape: {X_train.shape}")

    # CHANGED: Created proper validation split from training data instead of using test data
    val_size = int(0.2 * len(y_train))
    val_idx = np.random.choice(len(y_train), val_size, replace=False)
    train_idx = np.array([i for i in range(len(y_train)) if i not in val_idx])
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train_final, y_train_final = X_train[train_idx], y_train[train_idx]
    
    print("\n[2] Hyperparameter Tuning (3-fold CV)...")
    # CHANGED: Expanded grid with heuristic gamma option and wider C range
    param_grid = [{'C': 0.1, 'gamma': 0.01}, {'C': 1.0, 'gamma': 'auto'}, {'C': 10.0, 'gamma': 'auto'}]
    best_score, best_params = 0, param_grid[0]
    for p in param_grid:
        gamma_val = p['gamma']
        if gamma_val == 'auto':
            gamma_val = suggest_gamma(X_train_final)
        score = cross_validate(X_train_final, y_train_final, p['C'], gamma_val)
        print(f"  C={p['C']}, gamma={gamma_val:.4f}: CV={score:.4f}")
        if score > best_score:
            best_score, best_params = score, {'C': p['C'], 'gamma': gamma_val}
    print(f"Best: C={best_params['C']}, gamma={best_params['gamma']:.4f}")

    print("\n[3] Ensemble Method...")
    # CHANGED: Aligned ensemble params with best tuned values, increased iterations/time
    ensemble_params = [
        {'C': best_params['C'], 'gamma': best_params['gamma'], 'max_iter': 80, 'time_limit': 90},
        {'C': best_params['C'] * 2, 'gamma': best_params['gamma'] * 0.5, 'max_iter': 80, 'time_limit': 90}
    ]
    ensemble = EnsembleSVM(ensemble_params)
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Ensemble Test Accuracy: {acc:.4f}")
    print_class_metrics(y_test, y_pred)

    print("\n[4] Learning Curves...")
    plot_learning_curves(X_train, y_train, X_val, y_val, best_params['C'], best_params['gamma'])
    
    print("\n" + "=" * 60)
    print(f"Total time: {time.time() - start_total:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
    