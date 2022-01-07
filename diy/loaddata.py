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
train = pd.read_csv('../train.csv')
ss = pd.read_csv('../sample_submission.csv')
train_none = train[train.class_name == 'No finding']
reliable_annotators = ['R9']
train_reliable = train[train.rad_id.isin(reliable_annotators)]
train = train_reliable