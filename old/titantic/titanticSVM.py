import csv as csv
import numpy as np
from sklearn import svm
import os

data_dir = ('/Users/njchiang/CloudStation/kaggle/titantic')

csv_file_object = csv.reader(open(os.path.join(data_dir, 'train.csv'), 'rb')) 	# Load in the csv file
header = csv_file_object.next() 						# Skip the fist line as it is a header
data = [] 												# Create a variable to hold the data

for row in csv_file_object: 							# Skip through each row in the csv file,
    data.append(row[0:]) 								# adding each row to the data variable
data = np.array(data) 									# Then convert from a list to an array.

# Now I have an array of 12 columns and 891 rows
# I can access any element I want, so the entire first column would
# be data[0::,0].astype(np.float) -- This means all of the rows (from start to end), in column 0
# I have to add the .astype() command, because
# when appending the rows, python thought it was a string - so needed to convert

# clean out nans and replace them with means
def preprocess_train_features(d):
    d[d == ''] = np.nan
    d[d[:, 1] == 'female', 1] = 1
    d[d[:, 1] == 'male', 1] = 2
    d = d.astype(np.float)
    # clean out nans
    n = np.nanmean(d, axis = 0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(d))
    #Place column means in the indices. Align the arrays using take
    d[inds] = np.take(n, inds[1])
    return d, n

# essentially the same but replace nans with the mean from the training set
def preprocess_features(d, n):
    d[d == ''] = np.nan
    d[d[:, 1] == 'female', 1] = 1
    d[d[:, 1] == 'male', 1] = 2
    d = d.astype(np.float)
    # clean out nans
    #Find indicies that you need to replace
    inds = np.where(np.isnan(d))
    #Place column means in the indices. Align the arrays using take
    d[inds] = np.take(n, inds[1])
    return d


# let's train a classifier.
clf = svm.NuSVC(kernel='rbf')
labels = data[:, 1].astype(np.float)
# feature selection here...
# for now take class, sex, age, sibsp, parch
feat_idx = np.array([2, 4, 5, 6, 7])
# replace missing data with nan
features, nanmeans = preprocess_train_features(data[:, feat_idx].copy())

# train:
clf.fit(features, labels)

# First, read in test.csv
test_file = open(os.path.join(data_dir, 'test.csv'), 'rb')
test_file_object = csv.reader(test_file)
test_header = test_file_object.next()

# Also open the a new file so I can write to it. Call it something descriptive
# Finally, loop through each row in the train file, and look in column index [3] (which is 'Sex')
# Write out the PassengerId, and my prediction.
# This classifier decides based on whether they were female or male.

predictions_file = open(os.path.join(data_dir, 'rbfsvmmodel.csv'), "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])	# write the column headers
for row in test_file_object:                                    # For each row in test file,
    thisperson = np.array([row[i] for i in feat_idx-1]).reshape(1, len(nanmeans))
    predictions_file_object.writerow([row[0], str(
                                      np.round(clf.predict(preprocess_features(
                                          thisperson, nanmeans)))[0].astype(np.int8))])

test_file.close()												# Close out the files.
predictions_file.close()





