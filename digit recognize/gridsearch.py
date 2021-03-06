import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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
if __name__ == '__main__':
    # clf_obj = Pipeline([('pca',PCA()),('svm',SVC())])
    # c_range = [0.04, 0.045, 0.035]   #[0.01, 0.1, 1.0, 10.0, 100.0]#0.981845238095
    # n_features_range = [50]
    # parameters = [{'pca__n_components': n_features_range, 'svm__kernel': ['rbf'], 'svm__C': [10.0]}]
    # #parameters = [{'pca__n_components': n_features_range, 'svm__kernel': ['linear'], 'svm__C': c_range}, {
    # #    'pca__n_components': n_features_range, 'svm__kernel': ['rbf'], 'svm__C': c_range}]
    # gs = GridSearchCV(estimator=clf_obj, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=1)
    # gs = gs.fit(X_train, y_train)
    # print(gs.best_score_)
    # print(gs.best_params_)

    clf_obj = Pipeline([('pca', PCA(n_components=50)), ('svm', SVC(kernel='rbf', C=10.0, gamma=0.04))])
    clf_obj.fit(X,y)
    y_prd_pred = clf_obj.predict(X_prd)
    indics = np.arange(1, y_prd_pred.shape[0] + 1)
    y_prd_pred = np.row_stack((indics, y_prd_pred))
    prd_df = pd.DataFrame(columns=['ImageId', 'Label'], data=y_prd_pred.T)
    prd_df.to_csv('submission.csv', index=False)
# pca = PCA(n_components=100)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)
#
# svm = SVC()
# svm.fit(X_train_pca, y_train)
#
# y_train_pred = svm.predict(X_train_pca)
# acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
# print('Training accuracy: %.2f%%' % (acc * 100))
# y_test_pred = svm.predict(X_test_pca)
# acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
# print('Test accuracy: %.2f%%' % (acc * 100))