import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skimage.feature import hog

def load_mnist():
    # fetch mnist and normalize pixels
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist["data"], mnist["target"]
    X = X / 255.0

    # just want digit 8 vs the rest
    y = (y == '8').astype(int)

    # hog captures edge patterns — works better than raw pixels here
    hog_features = []
    for img in X:
        fd = hog(img.reshape(28, 28), orientations=9,
                 pixels_per_cell=(4, 4), cells_per_block=(3, 3))
        hog_features.append(fd)
    X = np.array(hog_features)

    # split into train(80%), validation(10%), test(10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test
def load_mnist_multiclass():
    # fetch mnist and normalize pixels
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist["data"], mnist["target"]
    X = X / 255.0

    # convert labels to integers
    y = y.astype(int)

    # hog captures edge patterns — works better than raw pixels here
    hog_features = []
    for img in X:
        fd = hog(img.reshape(28, 28), orientations=9,
                 pixels_per_cell=(4, 4), cells_per_block=(3, 3))
        hog_features.append(fd)
    X = np.array(hog_features)

    # split into train(80%), validation(10%), test(10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test