import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from neuralnet import NeuralNetMLP,MLPGradientCheck

def load_data():
    """Load MNIST data from `path`"""
    data = pd.read_csv('./Digit Recognizer/train.csv')
    images = data.iloc[:,1:]
    labels = data.iloc[:, [0]]
    return images, labels

X, y = load_data()
X, y = X.values, y.values.T.reshape(y.values.shape[0],)
print('Rows: %d, columns: %d' % (X.shape[0], X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# X_test, y_test = load_mnist('', kind='t10k')
# print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

img_prd = pd.read_csv('./Digit Recognizer/test.csv')
X_prd = img_prd.values

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
# np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
# np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
# np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')
#
# X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')
# y_train = np.genfromtxt('train_labels.csv', dtype=int, delimiter=',')
# X_test = np.genfromtxt('test_img.csv', dtype=int, delimiter=',')
# y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')

nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50, l2=0.1,
                  l1=0.0, epochs=1000, eta=0.001, alpha=0.001, decrease_const=0.00001,
                  shuffle=True, minibatches=50, random_state=1)
nn.fit(X_train, y_train, print_progress=True)
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()

batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]
plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (acc * 100))

y_prd_pred = nn.predict(X_prd)
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

nn_check = MLPGradientCheck(n_output=10, n_features=X_train.shape[1], n_hidden=10, l2=0.0,
                            l1=0.0, epochs=10, eta=0.001, alpha=0.0, decrease_const=0.0, minibatches=1, random_state=1)
nn_check.fit(X_train[:5], y_train[:5], print_progress=False)