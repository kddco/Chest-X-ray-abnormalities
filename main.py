import random
import cv2
import pydicom

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
from tqdm import tqdm

import matplotlib.pyplot as plt
from utilities_x_ray import read_xray,showXray

import numpy as np
# train = pd.read_csv('train.csv')
# ss = pd.read_csv('sample_submission.csv')
## 加載csv
train = pd.read_csv('D:/機器學習_胸部x光/train.csv')
ss = pd.read_csv('D:/機器學習_胸部x光/sample_submission.csv')
print(train.head())

# print(train.shape) #輸出檔案個數 和每列的colnum個數
# print(train.image_id.describe()) #
# print(train[train.image_id == '03e6ecfa6f6fb33dfeac6ca4f9b459c9']) #輸出這個id的資料
# print(train.class_name.value_counts())

plt.figure(figsize=(8,10))
plt.imshow(read_xray('d:/機器學習_胸部x光/train/03e6ecfa6f6fb33dfeac6ca4f9b459c9.dicom'),cmap=plt.cm.bone)
plt.show()
showXray('d:/機器學習_胸部x光/train/03e6ecfa6f6fb33dfeac6ca4f9b459c9.dicom',train,with_boxes=True)
