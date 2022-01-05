import pickle
from time import time
import numpy, os
from numpy.lib.index_tricks import nd_grid
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import cv2

#Path to the root image directory containing sub-directories of images
path="dataset/"


h = 50
w = 50
print("Image dimension after resize (h,w) :", h, w)

n_sample = 0  # Initial sample count
label_count = 0  # Initial label count
n_classes = 0  # Initial class count


##2

# Flat image Feature Vector
X = []
# Int array of Label Vector
Y = []

target_names = [] #Array to store the names of the persons

for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        img=cv2.imread(path+directory+"/"+file)
        img=cv2.resize(img, (w,h))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.GaussianBlur(img, (125, 125), 0)
        featurevector=numpy.array(img).flatten()
        X.append(featurevector)
        Y.append(label_count)
        n_sample = n_sample + 1
    target_names.append(directory)
    label_count=label_count+1

print("Samples :", n_sample)
print("Class :", target_names)
n_classes = len(target_names)

# Ghi data
pick_in = open('models/dataX.pickle','wb') #Write file
pickle.dump(X, pick_in)
pick_in.close()

with open('models/dataY.pickle', 'wb') as f: #Write file
    pickle.dump(Y, f)

with open('models/dataNames.pickle', 'wb') as f:
    pickle.dump(target_names, f)


# split into a training and teststing set

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

# print("Extracting the top %d eigenfaces from %d faces"
#       % (n_components, len(X_train)))

# vẽ biểu đô
t0 = time()

# pca = PCA().fit(X_train)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()



n_components = 250
pca = PCA(n_components=n_components,whiten=True).fit(X_train)

with open('models/pca.pickle', 'wb') as f:
    pickle.dump(pca, f)

print("done in %0.3fs" % (time() - t0))

# eigenfaces = pca.components_.reshape((n_components, h, w))

print("\n")
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))



import matplotlib.pyplot as plt

# plot = plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=y_train)
# plt.legend(handles=plot.legend_elements()[0], labels=list(target_names))
# plt.show()

##4 Show 7 ảnh sau PCA
# import matplotlib.pyplot as plt
#
# def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
#     """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#     for i in range(7):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i], cmap=plt.cm.gray)
#         plt.title(titles[i], size=12)
#         plt.xticks(())
#     plt.show()
#
#
# eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# plot_gallery(eigenfaces, eigenface_titles, h, w)

##5

###############################################################################
# Train a SVM classification model
print("\n")
print("Fitting the classifier to the training set")
t0 = time()

param_grid = {'C': [1,100,1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1, 10, 100, 1000], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=5)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print('Best score:', clf.best_score_)
print("\n")
print("Best estimator found by grid search : ")
print(clf.best_estimator_)

###############################################################################
# Quantitative evaluation of the model quality on the test set

print("\n")
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print(clf.score(X_test_pca,y_test))
print("done in %0.3fs" % (time() - t0))

print("\nClassification Report : ")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix : ")
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Save the model
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))



print(target_names)
