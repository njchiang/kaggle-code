import csv as csv
import numpy as np
from sklearn import svm
from scipy.linalg import svd
from sklearn.decomposition import PCA
import os

# data_dir = ('/Users/njchiang/CloudStation/kaggle/digits')
data_dir = 'D:\CloudStation\kaggle\digits'

# read in training data
csv_file_object = csv.reader(open(os.path.join(data_dir, 'train.csv'), 'rb')) 	# Load in the csv file
header = csv_file_object.next() 						# Skip the fist line as it is a header
data = [] 												# Create a variable to hold the data

for row in csv_file_object: 							# Skip through each row in the csv file,
    data.append(row[0:]) 								# adding each row to the data variable
data = np.array(data) 									# Then convert from a list to an array.


# define preprocessing here
# learning the raw images is probably a lot of work... so let's use principal components
# (maybe independent components later?)
# robust PCA algorithm for high dimensional data. rows = observations, cold = features...
# def efficientpca(x):
#     #demean x
#     mx = np.mean(x, axis=0)
#     cx = x - mx
#     vec, val = svd(np.dot(cx, cx.T))
#     return np.dot(cx.T, vec),
#
# # to accompany efficientpca
# def applypca(x, p):
#     return np.dot(x, np.dot(p, np.linalg.inv(np.dot(p.T,p))))

# check which dimension is larger-- number of obs or number of features
def preprocess_train(d, thr):
    # if d.shape[1] > d.shape[0]:
    #     efficientpca(d)
    p = PCA()
    p.fit(d)
    nf = np.sum(np.cumsum(p.explained_variance_ratio_) < thr)
    p = PCA(n_components=nf)
    return p.fit_transform(d), p.fit(d)


def preprocess_test(d, p):
    # for now this only works with low dimensional observations
    return p.transform(d)


# define apply preprocessing here
print "running pca"
train_features, pca = preprocess_train(data[:,1:].astype(np.float16), .8)
# define classifier
clf = svm.NuSVC(kernel='linear')
labels = data[:, 0].astype(np.float)
print "done"
# train:
print "training classifier... "
clf.fit(train_features, labels)
print "done"
# First, read in test.csv

# Then convert from a list to an array.

test_file = open(os.path.join(data_dir, 'test.csv'), 'rb')
test_file_object = csv.reader(test_file)
test_header = test_file_object.next()

tdata = []
for row in test_file_object:                                    # For each row in test file,
    # classify each entry here
    tdata.append(row[0:]) 								# adding each row to the data variable

test_file.close()												# Close out the files.


test_file = open(os.path.join(data_dir, 'test.csv'), 'rb')
test_file_object = csv.reader(test_file)
test_header = test_file_object.next()

# apparently this takes a long time, so applying PCA here
tdata = np.array(tdata).astype(np.float16)
test_features = preprocess_test(tdata, pca)
print "running classification"
res = clf.predict(test_features).astype(np.int)
print "done"

# open output file
predictions_file = open(os.path.join(data_dir, 'pcasvmmodel.csv'), "wb")
predictions_file_object = csv.writer(predictions_file)
# set up output... no output for this one
predictions_file_object.writerow(["label"])	# write the column headers
i = 0
for row in test_file_object:    # For each row in test file,
    # classify each entry here
    i += 1
    predictions_file_object.writerow([str[i], str(res[i-1])])

predictions_file.close()
test_file.close()												# Close out the files.
