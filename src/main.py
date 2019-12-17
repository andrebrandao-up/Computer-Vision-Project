# Required modules
import matplotlib.cm as cm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import skin_detector
import sys
import os
import pandas as pd
import age_classifier
import pickle
from os import listdir
from os.path import isfile, join
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def face_percentage_from_image(image, faces):
    return sum([w*h for (x, y, w, h) in faces])

def skin_percentage_from_image(image):
    # skin color limits in YCrCb color space.
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)
    # copy image.
    temp_img = image.copy()
    # convert to YCrCb.
    YCrCb_img = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    # copmute skin region.
    skinRegionYCrCb = cv2.inRange(YCrCb_img, min_YCrCb, max_YCrCb)/255
    height, width = temp_img.shape[:2]
    return sum(sum(skinRegionYCrCb))
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(skinRegionYCrCb, interpolation='bilinear', cmap=cm.Greys_r)
    plt.show()

def get_faces(image):
    # copy image.
    temp_img = image.copy()
    # convert to gray.
    gray_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    # create viola-jones classifier.
    haar_face_cascade = cv2.CascadeClassifier('../rude-carnie/haarcascade_frontalface_default.xml')
    # detect faces.
    return haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);

def is_child(face):
    return eval(age_classifier.get_age(face))[0] < 18

def get_stats(image):
    # skin mask.
    mask = skin_detector.process(image)
    # list with face positions.
    faces = get_faces(image)
    # No face then return.
    if len(faces) == 0: return 0, 0, []
    # get image in grb.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # get image size.
    height, width = image.shape[:2]
    # compute features
    face_pixels_number = face_percentage_from_image(image, faces)
    skin_pixels_number = mask
    skin_pixels_number = len(skin_pixels_number[skin_pixels_number>1])
    return skin_pixels_number/(height*width), skin_pixels_number/face_pixels_number, faces

def is_child_porn(image_path):
    # read image.
    image = cv2.imread(image_path)
    # get statistics about image.
    skin_total_ratio, skin_face_ratio, faces = get_stats(image)
    # no faces, then not child porn.
    if len(faces) == 0: return False
    # open pre-trained decision tree for porn detection.
    with open("models/dt_model.pkl", "rb") as model:
        clf = pickle.load(model)
    # classify as porn or not
    porn = clf.predict([[skin_total_ratio, skin_face_ratio]])[0] # NOT PORN = 0, PORN = 1
    if porn:
        for x, y, w, h in faces:
            if w < 50 or h < 50: continue
            crop_img = image[y:y+h+100, x-50:x+w+50]
            cv2.imwrite("cropped.png", crop_img)
            if is_child("cropped.png"):
                return True
    return False

if __name__ == "__main__":
    print(is_child_porn(sys.argv[1]))
