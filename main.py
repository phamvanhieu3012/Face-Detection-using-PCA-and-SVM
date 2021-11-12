import pickle

import numpy as np
import pandas as pd
import cv2
import time
import random

from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.feature import hog
import os
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# folderpath="./105_classes_pins_dataset/"
# cascade = "./haarcascade_frontalface_default.xml"
# height=128
# width=64
# data=[]
# labels=[]
# Celebs=[]

# Xử lý dataset
for dirname,_, filenames in tqdm(os.walk(folderpath)):
    for filename in filenames:
        image = cv2.imread(os.path.join(dirname, filename))
        image= cv2.resize(image , (width,height))
        labels.append(dirname.split("/")[-1])
        data.append(image)

le = LabelEncoder()
#Nhan ten
Labels= le.fit_transform(labels)
#Anh xam
data_gray = [cv2.cvtColor(data[i] , cv2.COLOR_BGR2GRAY) for i in range(len(data))]

#Nhan anh [0,1,2]
Labels = np.array(Labels).reshape(len(Labels),1)


#HOG
ppc =8
cb=4
hog_features=[]
hog_image=[]
for image in tqdm(data_gray):
    fd , hogim = hog(image , orientations=9 , pixels_per_cell=(ppc , ppc) , block_norm='L2' , cells_per_block=(cb,cb) , visualize=True)
    hog_image.append(hogim)
    hog_features.append(fd)

#SVM to fit
hog_features = np.array(hog_features)
df = np.hstack((hog_features,Labels))

#X train ...
X_train , X_test , Y_train , y_test = train_test_split(df[:,:-1] ,
                                                       df[:,-1],
                                                       test_size=0.3 ,
                                                       random_state=0 ,
                                                       stratify=df[:,-1])


#PCA
pca = PCA(n_components=150 , svd_solver='randomized' , whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#SVM
svc_model = SVC(kernel='rbf' , class_weight='balanced' , C=1000 , gamma=0.0082)
svc_model.fit(X_train_pca , Y_train)
print("Predicting the people names on the testing set")

# y_pred = clf.predict(X_test_pca)

#Train Accuracy
pred = svc_model.predict(X_train_pca)
train_acc = accuracy_score(Y_train, pred)
print("Training Accuracy: ", train_acc)

##Test Accuracy
pred = svc_model.predict(X_test_pca)
test_acc = accuracy_score(y_test, pred)
print("Test Accuracy: ", test_acc)

model_name = 'svm-{}.model'.format(str(int(test_acc*100)))
pickle.dump(svc_model, open(model_name, 'wb'))

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_pca, Y_train)



##Test

# img = cv2.imread('random/Dad.jpg')
# cv2.imshow("Anh: ",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
# labels= ["Adriana Lima", "Alex  Lawther", "Alexandra Daddario"]
#
# print(pred[0])
# person = labels[int(pred[0])] #labels[2]
# print("Is it {} ?".format(person))
#
# cv2.rectangle(img, (12, 12), (12, 12), (0, 0, 255), 2)
# cv2.putText(img, person, (12-10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
# cv2.imshow("Prediction: " + person + "?", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Test 2
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(img,labels[int(pred[0])],(x,y+h), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("Faces",img)
    if(cv2.waitKey(1) == ord('q')):
        break
    pass
cam.release()
cv2.destroyAllWindows()