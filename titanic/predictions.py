import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit

addpoly = True
plot_lc = 0

print('----- Loading the data -----')
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

print('train dataset {} and test dataset {}'.format(str(train_dataset.shape), str(test_dataset.shape)))
train_dataset.head()