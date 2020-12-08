import numpy as np
from data_util import imageNet_100
from model.svm import svm_loss_naive, svm_loss_vectorized, LinearSVM
import time

X_train, y_train, X_test, y_test = imageNet_100()

num_training = X_train.shape[0]
num_dev = 500
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

mean_image = np.mean(X_train, axis=0).astype(np.float16)
# second: subtract the mean image from train and test data

X_train -= mean_image
X_test -= mean_image
X_dev -= mean_image
X_dev = X_dev / len(X_dev)


# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_test.shape, X_dev.shape)

# generate a random SVM weight matrix of small numbers
W = np.random.randn(150529, 100) * 0.0001 # imageNet-100 class =100

#loss_naive, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005) #(500,3073)
#print('Naive loss: %f' % (loss_naive, ))
#loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
#print('Vectorized loss: %f' % (loss_vectorized, ))
#print('difference: %f' % (loss_naive - loss_vectorized))

svm = LinearSVM() # Make SVM Model
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                      num_iters=1500, verbose=True) # Train SVM
toc = time.time()
print('That took %fs' % (toc - tic))

y_train_pred = svm.predict(X_train)
print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))

y_test_pred = svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)