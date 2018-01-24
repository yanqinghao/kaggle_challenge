import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def load_data():
    """Load MNIST data from `path`"""
    data = pd.read_csv('./Digit Recognizer/train.csv')
    images = data.iloc[:,1:]
    labels = data.iloc[:, [0]]
    return images, labels

X, y = load_data()
X, y = X.values/255.0, y.values.T.reshape(y.values.shape[0],)
print('Rows: %d, columns: %d' % (X.shape[0], X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

img_prd = pd.read_csv('./Digit Recognizer/test.csv')
X_prd = img_prd.values/255.0

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

svm = SVC()
svm.fit(X_train_pca, y_train)

y_train_pred = svm.predict(X_train_pca)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))
y_test_pred = svm.predict(X_test_pca)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (acc * 100))