import numpy as np
import tensorflow as tf
import time
from skimage.feature import hog

class KernelSVM:
    def __init__(self, C=1.0, gamma=0.01, tol=1e-3):
        self.C = C
        self.gamma = gamma
        self.tol = tol
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
        print(f"Training Kernel SVM with {X.shape[0]} samples...")
        print(f"Parameters: C={self.C}, gamma={self.gamma}")
        
        self.X = X
        self.y = y.astype(np.float64)
        self.n = X.shape[0]
        
        print("Computing kernel matrix...")
        self.K = self.rbf_kernel(X, X)
        
        self.alpha = np.zeros(self.n)
        self.b = 0.0
        
        print("Running SMO optimization...")
        start_time = time.time()
        
        examine_all = True
        num_changed = 0
        iteration = 0
        
        while num_changed > 0 or examine_all:
            num_changed = 0
            
            if examine_all:
                for i in range(self.n):
                    num_changed += self.examine_example(i)
            else:
                for i in np.where((self.alpha > 0) & (self.alpha < self.C))[0]:
                    num_changed += self.examine_example(i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                sv = np.sum(self.alpha > 1e-5)
                print(f"  iter {iteration}: changed={num_changed}, SV={sv}, time={elapsed:.1f}s")
            
            if iteration > 500:
                print("  Max iterations reached")
                break
            
            if time.time() - start_time > 300:
                print("  Time limit reached")
                break
        
        print(f"Training completed in {time.time()-start_time:.1f}s")
        
        sv_idx = np.where(self.alpha > 1e-5)[0]
        print(f"Number of support vectors: {len(sv_idx)}")
    
    def examine_example(self, i):
        yi = self.y[i]
        Ei = self.compute_error(i)
        
        r = Ei * yi
        
        if r < -self.tol and yi == 1 and self.alpha[i] < self.C:
            return self.heuristic_select_j(i)
        if r < -self.tol and yi == -1 and self.alpha[i] > 0:
            return self.heuristic_select_j(i)
        if r > self.tol and yi == 1 and self.alpha[i] > 0:
            return self.heuristic_select_j(i)
        if r > self.tol and yi == -1 and self.alpha[i] < self.C:
            return self.heuristic_select_j(i)
        
        return 0
    
    def heuristic_select_j(self, i):
        Ei = self.compute_error(i)
        non_bound = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        
        if len(non_bound) > 1:
            max_diff = 0
            j = -1
            for k in non_bound:
                Ek = self.compute_error(k)
                diff = abs(Ei - Ek)
                if diff > max_diff:
                    max_diff = diff
                    j = k
            if j >= 0:
                if self.step(i, j):
                    return 1
        
        all_idx = np.arange(self.n)
        all_idx = all_idx[all_idx != i]
        np.random.shuffle(all_idx)
        
        for j in all_idx[:min(20, len(all_idx))]:
            if self.step(i, j):
                return 1
        
        return 0
    
    def step(self, i, j):
        if i == j:
            return 0
        
        yi, yj = self.y[i], self.y[j]
        Ei, Ej = self.compute_error(i), self.compute_error(j)
        
        ai_old, aj_old = self.alpha[i], self.alpha[j]
        
        if yi != yj:
            L = max(0, aj_old - ai_old)
            H = min(self.C, self.C + aj_old - ai_old)
        else:
            L = max(0, ai_old + aj_old - self.C)
            H = min(self.C, ai_old + aj_old)
        
        if abs(L - H) < 1e-8:
            return 0
        
        eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
        if eta >= 0:
            return 0
        
        aj_new = aj_old - yj * (Ei - Ej) / eta
        aj_new = np.clip(aj_new, L, H)
        
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
        
        self.alpha[i] = ai_new
        self.alpha[j] = aj_new
        self.b = b_new
        
        return 1
    
    def compute_error(self, i):
        return np.dot(self.alpha * self.y, self.K[:, i]) + self.b - self.y[i]
    
    def predict(self, X):
        K = self.rbf_kernel(X, self.X)
        decision = np.dot(K, self.alpha * self.y) + self.b
        return np.sign(decision)


def extract_hog_features(images):
    features = []
    for img in images:
        feat = hog(img, orientations=9, pixels_per_cell=(4, 4),
                   cells_per_block=(2, 2), feature_vector=True)
        features.append(feat)
    return np.array(features)


def load_mnist_data(n_train=2000, n_test=1000):
    print("Loading MNIST dataset...")
    
    (X_train_raw, y_train), (X_test_raw, y_test) = tf.keras.datasets.mnist.load_data()
    
    print("Extracting HOG features...")
    X_train = extract_hog_features(X_train_raw)
    X_test = extract_hog_features(X_test_raw)
    
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    n_per_class = n_train // 2
    
    X_8 = X_train[y_train == 8][:n_per_class]
    y_8 = np.ones(n_per_class)
    
    X_not_8 = X_train[y_train != 8][:n_per_class]
    y_not_8 = -np.ones(n_per_class)
    
    X_tr = np.vstack([X_8, X_not_8])
    y_tr = np.concatenate([y_8, y_not_8])
    
    n_test_per = n_test // 2
    
    X_test_8 = X_test[y_test == 8][:n_test_per]
    y_test_8 = np.ones(n_test_per)
    
    X_test_not = X_test[y_test != 8][:n_test_per]
    y_test_not = -np.ones(n_test_per)
    
    X_te = np.vstack([X_test_8, X_test_not])
    y_te = np.concatenate([y_test_8, y_test_not])
    
    idx = np.random.permutation(len(y_tr))
    X_tr, y_tr = X_tr[idx], y_tr[idx]
    
    idx = np.random.permutation(len(y_te))
    X_te, y_te = X_te[idx], y_te[idx]
    
    print(f"Training: {len(y_tr)} (8: {int(np.sum(y_tr==1))}, not 8: {int(np.sum(y_tr==-1))})")
    print(f"Test: {len(y_te)} (8: {int(np.sum(y_te==1))}, not 8: {int(np.sum(y_te==-1))})")
    
    return (X_tr, y_tr), (X_te, y_te)


def compute_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    return int(TP), int(TN), int(FP), int(FN)


def main():
    print("=" * 60)
    print("  Kernel SVM for Digit 8 Classification")
    print("=" * 60)
    
    np.random.seed(42)
    
    (X_train, y_train), (X_test, y_test) = load_mnist_data(n_train=1000, n_test=500)
    
    svm = KernelSVM(C=1.0, gamma=0.001, tol=1e-3)
    svm.fit(X_train, y_train)
    
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    
    y_train_pred = svm.predict(X_train)
    train_acc = np.mean(y_train_pred == y_train)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    
    y_test_pred = svm.predict(X_test)
    test_acc = np.mean(y_test_pred == y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    TP, TN, FP, FN = compute_confusion_matrix(y_test, y_test_pred)
    
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              8      Not 8")
    print(f"Actual  8     {TP:4d}      {FN:4d}")
    print(f"        Not 8 {FP:4d}      {TN:4d}")
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\n" + "=" * 60)
    print("  Implementation Summary")
    print("=" * 60)
    print(f"- Algorithm: Sequential Minimal Optimization (SMO)")
    print(f"- Kernel: RBF/Gaussian K(x,x') = exp(-gamma*||x-x'||^2)")
    print(f"- Regularization parameter C: {svm.C}")
    print(f"- Kernel parameter gamma: {svm.gamma}")
    print(f"- Features: {X_train.shape[1]} (HOG features from 28x28 images)")
    print("=" * 60)


if __name__ == "__main__":
    main()