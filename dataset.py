import sys
import os
from collections import defaultdict
import numpy as np
import cv2


def dataset(base_dir):
    d = defaultdict(list)

    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("\\")
            label = suffix.split("\\")[0]
            #print(label)
            if label == "1" or label == "0":
                d[label].append(file_path)

    tags = sorted(d.keys())
    #print("classes : {}".format(tags))
    #print(d[".DS_Store"])

    X = []
    y = []
    fname = []

    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            img = cv2.imread(filename)
            #img = img[:, :, :3] # remove alpha
            if img is not None:
                height, width, chan = img.shape

                assert chan == 3

                name = filename.split("\\")[2]
                name = name.split("_")[0][1:]
                fname.append(name)
                X.append(img)
                y.append(class_index)

    X = np.array(X).astype(np.float32)
    y = np.array(y)
    fname = np.array(fname)

    print("fname : {}\nX : {}\ny : {}\ntags : {}\n".format(fname,X,y,len(tags)))

    return fname, X, y, tags

dataset("dataset/dataset_120_50/testing")
