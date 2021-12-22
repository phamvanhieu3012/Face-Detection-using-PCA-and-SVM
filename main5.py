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
import cv2

#Path to the root image directory containing sub-directories of images
path="dataset/"

# data_slice = [70,195,78,172] # [ ymin, ymax, xmin, xmax]
# resize_ratio = 2.5
# h = int((data_slice[1] - data_slice[0])/resize_ratio) #ymax - ymin slice, Height of image in float
# w = int((data_slice[3] - data_slice[2])/resize_ratio) #xmax - xmin slice, Width of image in float
h = 50
w = 50
print("Image dimension after resize (h,w) :", h, w)

n_sample = 0 #Initial sample count
label_count = 0 #Initial label count
n_classes = 0 #Initial class count

#PCA Component
# n_components = 242

##2

#Flat image Feature Vector
X=[]
#Int array of Label Vector
Y=[]

target_names = [] #Array to store the names of the persons

for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        img=cv2.imread(path+directory+"/"+file)
        img=cv2.resize(img, (w,h))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

##3 PCA

###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and teststing set

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction


# print("Extracting the top %d eigenfaces from %d faces"
#       % (n_components, len(X_train)))

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

t0 = time()
pca = PCA(n_components=0.8, whiten=True)
pca = Pipeline([('scaler', StandardScaler()), ('pca', pca)]).fit(X_train)

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

plot = plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=y_train)
plt.legend(handles=plot.legend_elements()[0], labels=list(target_names))
plt.show()

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
param_grid = {'C': [1,100,1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1, 10, 100], }
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

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# some time later...

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)

# print(loaded_model)

##6 Test

###############################################################################
# Prediction of user based on the model

# test = []
# testImage = "test/selena.jpg"
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# testImage=cv2.imread(testImage)[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
# testImage=cv2.resize(testImage, (w,h))
# testImage=cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
# testImageFeatureVector=numpy.array(testImage).flatten()
# test.append(testImageFeatureVector)
# testImagePCA = pca.transform(test) 	# Apply dimensionality reduction to test.
# testImagePredict=clf.predict(testImagePCA)
#
# print ("Predicted Name : " + target_names[testImagePredict[0]])
#
# ## Xuat anh
#
# testImage3 = cv2.imread("test/selena.jpg")
# testImage2 =cv2.resize(testImage3, (400,250))
# gray = cv2.cvtColor(testImage2, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# for (x, y, w, h) in faces:
#     cv2.rectangle(testImage2, (x, y), (x + w, y + h), (255, 0, 0), 3)
#     cv2.putText(testImage2, target_names[testImagePredict[0]], (x + x // 10, y + h + 20), \
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
# cv2.imshow('img', testImage2)
# cv2.waitKey()
#print('Accuracy: '+clf.score(testImagePCA, target_names))

print(target_names)

path_testdata = "test2/"
for file in os.listdir(path_testdata):
    test = []

    filename = path_testdata + file
    testImage = cv2.imread(filename)
    testImage = cv2.resize(testImage, (w, h))
    testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(testImage, (125, 125), 0)
    division = cv2.divide(testImage, smooth, scale=255)
    testImageFeatureVector = numpy.array(division).flatten()

    test.append(testImageFeatureVector)
    testImagePCA = pca.transform(test)
    testImagePredict = clf.predict(testImagePCA)

    print("File Source : " + filename)
    print("Predicted Name : " + target_names[testImagePredict[0]] + "\n")

##8 Mở cam test

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap.set(3,640)
cap.set(4,480)

while(True):
    # Capture frame-by-frame
    test = []
    face = []
    ret, frame = cap.read()
    xv, yv, cv = frame.shape
    if ret == True :
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, wf, hf) in faces:
            cy, cx = y + (hf // 2), x + (wf // 2)
            max_len = max(max(hf // 2, wf // 2), 125)

            if (x - max_len) <= 0 or (x + max_len) >= xv or (y - max_len) <= 0 or (y + max_len) >= yv:
                continue

            face_crop = (frame[cy - max_len:cy + max_len, cx - max_len:cx + max_len])
            # face_crop = face_crop[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]

            testImage = cv2.resize(face_crop, (w, h))
            cv2.imshow('face', testImage)

            testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
            testImageFeatureVector = numpy.array(testImage).flatten()
            test.append(testImageFeatureVector)
            testImagePCA = pca.transform(test)
            testImagePredict = clf.predict(testImagePCA)

            # create box on detected face
            frame = cv2.rectangle(frame, (x, y), (x + wf, y + hf), (255, 0, 0), 1)
            frame = cv2.rectangle(frame, (x, y + hf), (x + wf, y + hf + 30), (255, 0, 0), -1)
            # print label name on image
            cv2.putText(frame, "Name : " + target_names[testImagePredict[0]], (x + x // 10, y + hf + 20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

## 9 input video

# cap = cv2.VideoCapture("test/avil.mp4")
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# cap.set(3,640)
# cap.set(4,480)
#
# while cap.isOpened():
#     # Capture frame-by-frame
#     test = []
#     face = []
#     _, frame = cap.read()
#     xv, yv, cv = frame.shape
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, wf, hf) in faces:
#         cy, cx = y + (hf // 2), x + (wf // 2)
#         max_len = max(max(hf // 2, wf // 2), 125)
#
#         if (x - max_len) <= 0 or (x + max_len) >= xv or (y - max_len) <= 0 or (y + max_len) >= yv:
#             continue
#         face_crop = (frame[cy - max_len:cy + max_len, cx - max_len:cx + max_len])
#         face_crop = face_crop[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
#
#         testImage = cv2.resize(face_crop, (w, h))
#         cv2.imshow('face', testImage)
#
#         testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
#         testImageFeatureVector = numpy.array(testImage).flatten()
#         test.append(testImageFeatureVector)
#         testImagePCA = pca.transform(test)
#         testImagePredict = clf.predict(testImagePCA)
#
#         # create box on detected face
#         frame = cv2.rectangle(frame, (x, y), (x + wf, y + hf), (255, 0, 0), 1)
#         frame = cv2.rectangle(frame, (x, y + hf), (x + wf, y + hf + 30), (255, 0, 0), -1)
#         # print label name on image
#         cv2.putText(frame, "Name : " + target_names[testImagePredict[0]], (x + x // 10, y + hf + 20), \
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()