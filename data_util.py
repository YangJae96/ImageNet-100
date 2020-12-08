import numpy as np
import time
import os
import cv2
from config import cfg
from glob import glob

def get_test():
    X_test = []
    y_test = []

    test_list = os.listdir(cfg.test_dir)
    test_list.sort()
    cnt=0
    for class_label in test_list:
        for img_path in glob(os.path.join(cfg.test_dir, class_label, '*.JPEG')):
            cvimg = cv2.imread(img_path)
            X_test.append(cvimg)
            y_test.append(int(class_label)-1)
            cnt+=1
    print("hello2")

    x_te = np.concatenate(X_test).reshape(cnt,224,224,3).astype(np.float16)
    y_te = np.array(y_test)


    return x_te, y_te


def get_train():
    X_train = []
    y_train = []

    train_list = os.listdir(cfg.train_dir)
    train_list.sort()
    cnt=0

    for class_label in train_list:

        class_per_label=0
        for img_path in glob(os.path.join(cfg.train_dir, class_label, '*.JPEG')):
            cvimg = cv2.imread(img_path)
            X_train.append(cvimg)
            y_train.append(int(class_label)-1)
            cnt+=1
            class_per_label += 1
            if class_per_label==30:
                break

    print("hello1")
    x_tr = np.concatenate(X_train)
    x_tr = x_tr.reshape(cnt, 224, 224, 3).astype(np.float16)
    y_tr = np.array(y_train)
    return x_tr, y_tr


def imageNet_100():

    x_tr=0
    y_tr=0
    x_tr, y_tr = get_train()
    x_te, y_te = get_test()
    print("hello")
    return x_tr, y_tr, x_te, y_te
