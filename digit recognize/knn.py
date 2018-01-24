import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i in range(25):
#     img = X_train[y_train == 7][i].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
X_prd_pca = pca.transform(X_prd)

knn = KNeighborsClassifier(n_neighbors=5, p=3, metric='minkowski')
knn.fit(X_train_pca, y_train)

y_train_pred = knn.predict(X_train_pca)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))
y_test_pred = knn.predict(X_test_pca)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (acc * 100))

y_prd_pred = knn.predict(X_prd_pca)
indics = np.arange(1,y_prd_pred.shape[0] + 1)
y_prd_pred = np.row_stack((indics,y_prd_pred))
prd_df = pd.DataFrame(columns=['ImageId','Label'], data=y_prd_pred.T)
prd_df.to_csv('submission.csv',index=False)

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab= y_test_pred[y_test != y_test_pred][:25]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
