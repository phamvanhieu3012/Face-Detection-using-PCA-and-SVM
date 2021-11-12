import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classifier

###1

# dir = './105_classes_pins_dataset'
#
# categories = ['Adriana Lima', 'Alex Lawther', 'Alexandra Daddario']
#
# data = []
#
# for category in categories:
#     path = os.path.join(dir, category)  # Tao duong dan vao 2 thu muc
#     label = categories.index(category)  # [0,1]
#
#     for img in os.listdir(path):  # Xet cac list anh trong path
#         imgpath = os.path.join(path, img)
#         human_img = cv2.imread(imgpath, 0)
#         # cv2.imshow("Image",pet_img)
#         try:
#             human_img = cv2.resize(human_img, (140, 80))
#             image = np.array(human_img).flatten()  # Trai ra 1 duong thang
#
#             data.append([image, label])
#         except Exception as ex:
#             pass
#
# print(len(data))
#
# pick_in = open('data.pickle','wb') #Write file
# pickle.dump(data,pick_in)
# pick_in.close()

###2

pick_in = open('data.pickle', 'rb')  # Read file
data = pickle.load(pick_in)
pick_in.close()
#
random.shuffle(data)
features = []
labels = []
#
# # data = [ [[1,2,3,4],0],[[2,3,5,4],0] ]
# # feature = [1,2,3,4]
# # label = 0
for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.01)

print(ytest)
# model = SVC(C=1, kernel='poly', gamma='auto')
# model.fit(xtrain, ytrain)  # xtrain =  features , ytrain = labels
#
# # Ghi model
# pick = open('medel.sav','wb')
# pickle.dump(model,pick)
# pick.close()

# ###3
# # Đọc model
pick = open('medel.sav', 'rb')
model = pickle.load(pick)
pick.close()
#
# Truoc khi predic thi tao ra model train truoc

anhtest = cv2.imread('anh-test.jpg', 0)
anhtest = cv2.resize(anhtest, (140, 80))
imageTest = np.array(anhtest).flatten()  # Trai ra 1 duong thang
anhtest2 = cv2.imread('anhtest.jpg', 0)
anhtest2 = cv2.resize(anhtest, (140, 80))
imageTest2 = np.array(anhtest).flatten()
# cv2.imshow('Image',anhtest)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(xtest)
# print([imageTest,imageTest2])

prediction = model.predict(xtest)
# accuracy = model.score(xtest, ytest)

categories = ['Adriana Lima', 'Alex Lawther', 'Alexandra Daddario']

# kha nang
# print('Accuracy: ', accuracy)

print('Prediction is: ', categories[prediction[0]])

# Ảnh để test (lấy từ xtest)
img_test = xtest[0].reshape(140, 80)
plt.imshow(img_test, cmap='gray')
plt.show()
