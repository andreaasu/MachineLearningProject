import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import json
from scipy.ndimage import zoom
from tensorflow.keras.datasets import mnist
from skimage.feature import hog
from joblib import Parallel, delayed

CONFIG = {
    "target_digit": 8,
    "cell_size": 3,
    "feature": "hog",
    "decision_threshold": 2.0
}


def load_raw_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(float)
    X_test = X_test.reshape(-1, 784).astype(float)
    y_train = (y_train == CONFIG["target_digit"]).astype(int)
    y_test = (y_test == CONFIG["target_digit"]).astype(int)
    return X_train, y_train, X_test, y_test


def crop_bounding_box(img):
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img[rmin:rmax+1, cmin:cmax+1]
    target_size = 20
    h, w = cropped.shape
    if h > target_size or w > target_size:
        scale = target_size / max(h, w)
        cropped = zoom(cropped, scale)
    return cropped


def center_digit(img):
    if np.sum(img) == 0:
        return img
    y_coords, x_coords = np.where(img > 0)
    cy, cx = np.mean(y_coords), np.mean(x_coords)
    h, w = img.shape
    oy, ox = h // 2 - cy, w // 2 - int(cx)
    shifted = np.roll(img, int(oy), axis=0)
    shifted = np.roll(shifted, int(ox), axis=1)
    return shifted


def normalize_pixels(X):
    return X / 255.0


def extract_hog(X, cell_size=4):
    X_reshaped = X.reshape(-1, 28, 28).astype(float) / 255.0
    
    def process_img(img):
        return hog(img, orientations=9, pixels_per_cell=(cell_size, cell_size),
                  cells_per_block=(2, 2), block_norm='L2-Hys')
    
    features = Parallel(n_jobs=-1, verbose=1)(delayed(process_img)(img) for img in X_reshaped)
    return np.array(features)


def extract_zones(X, grid_size):
    n_samples = X.shape[0]
    zone_size = 28 // grid_size
    X_reshaped = X.reshape(n_samples, 28, 28)
    zones = X_reshaped.reshape(n_samples, grid_size, zone_size, grid_size, zone_size)
    features = zones.sum(axis=(2, 4))
    return features.reshape(n_samples, -1)


def preprocess(X):
    if CONFIG.get("feature") == "hog":
        X_hog = extract_hog(X, CONFIG.get("cell_size", 4))
        return (X_hog > 0.1).astype(int)
    X = X.reshape(-1, 28, 28)
    X_centered = np.array([center_digit(img) for img in X])
    X_bin = (X_centered > CONFIG["pixel_threshold"]).astype(int)
    X_zones = extract_zones(X_bin, CONFIG["grid_size"])
    X_zones = (X_zones >= CONFIG["zone_threshold"]).astype(int)
    return X_zones


class BernoulliNaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.likelihoods = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.likelihoods[c] = (np.sum(X_c, axis=0) + 1) / (len(X_c) + 2)

    def predict(self, X, decision_threshold=0.0):
        predictions = []
        confidence_scores = []
        for x in X:
            class_scores = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(
                    x * np.log(self.likelihoods[c]) +
                    (1 - x) * np.log(1 - self.likelihoods[c])
                )
                class_scores[c] = prior + likelihood
            confidence_scores.append(class_scores)
            predictions.append(max(class_scores, key=class_scores.get))

        if decision_threshold != 0.0:
            adjusted = []
            for cs in confidence_scores:
                diff = cs.get(1, float('-inf')) - cs.get(0, float('-inf'))
                adjusted.append(1 if diff > decision_threshold else 0)
            return np.array(adjusted)
        return np.array(predictions)

    def save(self, filepath):
        model_data = {
            "classes": self.classes.tolist(),
            "priors": {str(k): v for k, v in self.priors.items()},
            "likelihoods": {str(k): v.tolist() for k, v in self.likelihoods.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        model = cls()
        model.classes = np.array(model_data["classes"])
        model.priors = {int(k): v for k, v in model_data["priors"].items()}
        model.likelihoods = {int(k): np.array(v) for k, v in model_data["likelihoods"].items()}
        return model


def main():
    d = CONFIG["target_digit"]
    print(f"Detecting digit: {d}")
    print(f"Using HOG features, cell_size={CONFIG['cell_size']}, decision_threshold={CONFIG['decision_threshold']}")

    X_train, y_train, X_test, y_test = load_raw_data()
    X_train_feat = preprocess(X_train)
    X_test_feat = preprocess(X_test)

    if os.path.exists("detect_8_model.json"):
        print("Loading model...")
        model = BernoulliNaiveBayes.load("detect_8_model.json")
    else:
        print("Training model...")
        model = BernoulliNaiveBayes()
        model.fit(X_train_feat, y_train)
        model.save("detect_8_model.json")
        print("Model saved: detect_8_model.json")

    preds = model.predict(X_test_feat, decision_threshold=CONFIG["decision_threshold"])
    accuracy = np.mean(preds == y_test)

    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    tn = np.sum((preds == 0) & (y_test == 0))

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1 = float(f1)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"         Predicted not-8  Predicted 8")
    print(f"Actual not-8   {tn:5d}          {fp:5d}")
    print(f"Actual 8      {fn:5d}          {tp:5d}")


if __name__ == "__main__":
    main()