import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from imblearn.under_sampling import RandomUnderSampler

# loading the training and validation data into train_data/label and test_data/label
train = pd.read_csv('training.csv', sep='[,;]', header=None,  skiprows=1)
test = pd.read_csv('validation.csv', sep='[,;]', header=None,  skiprows=1)

train_data = np.array(train.iloc[:, 0:-1])
train_label = np.array(train.iloc[:, -1])

test_data = np.array(test.iloc[:, 0:-1])
test_label = np.array(test.iloc[:, -1])


def is_number(s):  # shows if s is a number
    try:
        float(s)
        return True
    except ValueError:
        return False


def most_frequent(arr):  # outputs the mode of a list of strings
    counter = 0
    num = arr[0]

    for i in arr:
        curr_frequency = np.count_nonzero(arr == i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


"""

preprocessing of train_data/label

I assumed that the data is delimited by both ; and ,  ...so we have 21 features and a class label

we first define how each column is to be treated (number or string) then we shift the columns based 
on where they should belong
we then change all strings into numbers and all nans are substituted by the mode of the column( string)
or its mean(numbers)

we then change the class labels into numbers

"""
for j in range(21):

    if is_number(train_data[0, j]) and (str(test_data[0, j]) != 'nan'):
        column_type = 'number'
    elif str(train_data[0, j]) == 'nan':
        if is_number(train_data[1, j]):
            column_type = 'number'
        else:
            column_type = 'char'
    else:
        column_type = 'char'

    numeric_column_arr = []
    if column_type == 'number':
        for k in train_data[:, j]:
            if is_number(k) and str(k) != 'nan' and str(train_data[i, j]) != 'NA':
                numeric_column_arr.append(k)
        mode_or_mean = np.mean(np.array(numeric_column_arr).astype(np.float))
    else:
        mode_or_mean = most_frequent(train_data[:, j])

    for i in range(len(train_data)):
        if str(train_data[i, j]) == 'nan' or str(train_data[i, j]) == 'NA':
            train_data[i, j] = mode_or_mean
            continue
        elif (is_number(train_data[i, j]) and column_type == 'number') or (
                not (is_number(train_data[i, j])) and column_type == 'char'):
            continue
        else:
            train_label[i] = train_data[i, -1]
            for k in range(1, 21 - j):
                train_data[i, -k] = train_data[i, -k - 1]
            train_data[i, j] = mode_or_mean

    if column_type == 'char':
        enumeration = np.array(list(np.ndenumerate(np.unique(train_data[:, j]))))[:, 1]
        for i in range(len(train_data)):
            train_data[i, j] = np.where(enumeration == train_data[i, j])[0][0]

for i in range(len(train_label)):
    if str(train_label[i]) == '"yes."':
        train_label[i] = 1
    elif str(train_label[i]) == '"no."':
        train_label[i] = 0


# preprocessing of test_data/label
# same as train_data/label
for j in range(21):

    if is_number(test_data[0, j]) and (str(test_data[0, j]) != 'nan'):
        column_type = 'number'
    elif str(test_data[0, j]) == 'nan':
        start = 0
        while str(test_data[start, j]) == 'nan':
            start += 1
        if is_number(test_data[start, j]):
            column_type = 'number'
        else:
            column_type = 'char'
    else:
        column_type = 'char'

    numeric_column_arr = []
    if column_type == 'number':
        for k in test_data[:, j]:
            if is_number(k) and str(k) != 'nan' and str(test_data[i, j]) != 'NA':
                numeric_column_arr.append(k)
        mode_or_mean = np.mean(np.array(numeric_column_arr).astype(np.float))
    else:
        mode_or_mean = most_frequent(test_data[:, j])

    for i in range(len(test_data)):
        if str(test_data[i, j]) == 'nan' or str(test_data[i, j]) == 'NA':
            test_data[i, j] = mode_or_mean
            continue
        elif (is_number(test_data[i, j]) and column_type == 'number') or (
                not (is_number(test_data[i, j])) and column_type == 'char'):
            continue
        else:
            test_label[i] = test_data[i, -1]
            for k in range(1, 21 - j):
                test_data[i, -k] = test_data[i, -k - 1]
            test_data[i, j] = mode_or_mean

    if column_type == 'char':
        enumeration = np.array(list(np.ndenumerate(np.unique(test_data[:, j]))))[:, 1]
        for i in range(len(test_data)):
            test_data[i, j] = np.where(enumeration == test_data[i, j])[0][0]

for i in range(len(test_label)):
    if str(test_label[i]) == '"yes."':
        test_label[i] = 1
    elif str(test_label[i]) == '"no."':
        test_label[i] = 0

"""

we notice that variable 19 in training data is the same as the output which isn't true for the validation set and leads 
to overfitting so when training i ignored it

we also notice that the number of yes class in training data is way bigger than no class which isn't true for the 
validation data so i u used random under sampling algorithm to resample the data for balanced class distribution and
also to reduce overfiiting

then i used a decision tree to fit the data and try the validation set


"""


rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(train_data[:, :-1].astype(np.float), train_label.astype(np.float))
clf = tree.DecisionTreeClassifier()
clf.fit(X_res, y_res)
score = clf.score(test_data[:, :-1].astype(np.float), test_label.astype(np.float))
print(score)
