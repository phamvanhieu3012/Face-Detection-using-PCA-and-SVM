from time import time
import numpy, os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import cv2
import pickle

#Path to the root image directory containing sub-directories of images
path="dataset/"

data_slice = [70,195,78,172] # [ ymin, ymax, xmin, xmax]
resize_ratio = 2.5
h = int((data_slice[1] - data_slice[0])/resize_ratio) #ymax - ymin slice, Height of image in float
w = int((data_slice[3] - data_slice[2])/resize_ratio) #xmax - xmin slice, Width of image in float
print("Image dimension after resize (h,w) :", h, w)

# n_sample = 0 #Initial sample count
# label_count = 0 #Initial label count
# n_classes = 0 #Initial class count

#PCA Component
n_components = 7

##2 D

#Flat image Feature Vector
X=[]
#Int array of Label Vector
Y=[]

target_names = [] #Array to store the names of the persons

# for directory in os.listdir(path):
#     for file in os.listdir(path+directory):
#         img=cv2.imread(path+directory+"/"+file)
#         img=cv2.resize(img, (w,h))
#         img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         featurevector=numpy.array(img).flatten()
#         X.append(featurevector)
#         Y.append(label_count)
#         n_sample = n_sample + 1
#     target_names.append(directory)
#     label_count=label_count+1

# print("Samples :", n_sample)
# print("Class :", target_names)
# n_classes = len(target_names)

# Ghi data
# pick_in = open('models/dataX.pickle','wb') #Write file
# pickle.dump(X, pick_in)
# pick_in.close()
#
# with open('models/dataY.pickle', 'wb') as f: #Write file
#     pickle.dump(Y, f)
#
# with open('models/dataNames.pickle', 'wb') as f:
#     pickle.dump(target_names, f)

# Đọc data
pick_in = open('models/dataX.pickle', 'rb')  # Read file
X = pickle.load(pick_in)
pick_in.close()

with open('models/dataY.pickle', 'rb') as f:
    Y = pickle.load(f)

with open('models/dataNames.pickle', 'rb') as f:
    target_names = pickle.load(f)


with open('models/pca.pickle', 'rb') as f:
    pca = pickle.load(f)


filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

###############################################################################
# Prediction of user based on the model


# # Xuat anh
#
# testImage3 = cv2.imread("test/selena.jpg")
# testImage2 =cv2.resize(testImage3, (200, 300))
# gray = cv2.cvtColor(testImage2, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# for (x, y, w, h) in faces:
#     cv2.rectangle(testImage2, (x, y), (x + w, y + h), (255, 0, 0), 3)
#     cv2.putText(testImage2, target_names[testImagePredict[0]], (x + x // 10, y + h + 20), \
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
# cv2.imshow('img', testImage2)
# cv2.waitKey()

# filename = 'finalized_model.sav'
# clf = pickle.load(open(filename, 'rb'))
#
# print(clf)


path_testdata = "test2/"
for file in os.listdir(path_testdata):
    test = []

    filename = path_testdata + file
    testImage = cv2.imread(filename)[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
    testImage = cv2.resize(testImage, (w, h))
    testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    testImageFeatureVector = numpy.array(testImage).flatten()

    test.append(testImageFeatureVector)
    testImagePCA = pca.transform(test)
    testImagePredict = loaded_model.predict(testImagePCA)

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
            face_crop = face_crop[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]

            testImage = cv2.resize(face_crop, (w, h))
            cv2.imshow('face', testImage)

            testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
            testImageFeatureVector = numpy.array(testImage).flatten()
            test.append(testImageFeatureVector)
            testImagePCA = pca.transform(test)
            testImagePredict = loaded_model.predict(testImagePCA)



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

# cap = cv2.VideoCapture("test/selena.mp4")
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