import csv as csv
import numpy as np
from sklearn import svm
import os

# specify paths
data_dir = ('/Users/njchiang/CloudStation/kaggle/digits')
data_dir = 'D:\CloudStation\kaggle\digits'

# read in training data
csv_file_object = csv.reader(open(os.path.join(data_dir, 'train.csv'), 'rb')) 	# Load in the csv file
header = csv_file_object.next() 						# Skip the fist line as it is a header
data = [] 												# Create a variable to hold the data

for row in csv_file_object: 							# Skip through each row in the csv file,
    data.append(row[0:]) 								# adding each row to the data variable
data = np.array(data) 									# Then convert from a list to an array.

# define preprocessing here

# define apply preprocessing here

# define classifier and labels
clf = svm.NuSVC(kernel='linear')
labels = data[:, 1].astype(np.float)

# train:
clf.fit(features, labels)

# First, read in test.csv
test_file = open(os.path.join(data_dir, 'test.csv'), 'rb')
test_file_object = csv.reader(test_file)
test_header = test_file_object.next()

# open output file, name it accordingly
predictions_file = open(os.path.join(data_dir, 'rbfsvmmodel.csv'), "wb")
predictions_file_object = csv.writer(predictions_file)
# set up output... no output for this one
# predictions_file_object.writerow(["PassengerId", "Survived"])	# write the column headers
for row in test_file_object:                                    # For each row in test file,
    # classify each entry here

test_file.close()												# Close out the files.
predictions_file.close()