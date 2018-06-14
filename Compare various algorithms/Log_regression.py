import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# loading the data from a csv file
raw_data = pd.read_csv("animals.csv")


# see first five rows of data
raw_data.head(n=5)

# Slicing the data into train and test sets
# training_data = raw_data.iloc[0:4000, :]
# testing_data = raw_data.iloc[4001:5134, :]

# summary Statistics of all the attributes in the data
raw_data.describe()

# removing attributes that will have very little or no effect on classifier
del raw_data["speed_min"]
del raw_data["acc_mean"]
del raw_data["acc_std"]

# creating a dummy variable from class attribute where DEER = 0, CATTLE = 1, ELK =2
# y = []
# for i in raw_data['class']:
#    if i == "DEER":
#        y.append(0)
#   elif i == "CATTLE":
#        y.append(1)
#    else:
#        y.append(2)
# print y

# binarification of "class" attribute into three dummmy variables
is_deer = [1 if raw_data["class"][i] == "DEER" else 0 for i in range(len(raw_data["class"]))]
is_cattle = [1 if raw_data["class"][i] == "CATTLE" else 0 for i in range(len(raw_data["class"]))]
is_elk = [1 if raw_data["class"][i] == "ELK" else 0 for i in range(len(raw_data["class"]))]

# conversion of data into NumPy array
data = raw_data.values
X = np.array(data[:, :22])

def estimator(data, which_class, target_vals):
    print "----------------------------For %s or not %s------------------------------" % (which_class, which_class)
    # Splitting the data into train and test data sets
    X_train, X_test, y_train_class, y_test_class = train_test_split(data, target_vals, random_state=10)

    # Training the classifier
    LRclf = LogisticRegression()
    LRclf.fit(X_train, y_train_class)

    print "Running classifier on train data"
    predicted_train = LRclf.predict(X_train)
    print "Confusion Matrix on train data : \n", confusion_matrix(y_train_class, predicted_train)
    accuracy_class_train = accuracy_score(y_train_class, predicted_train)
    print "Accuracy for %s on train data is : %f" % (which_class, accuracy_class_train), "\n\n"

    print "Running classifier on test data"
    predicted = LRclf.predict(X_test)
    print "Confusion matrix : \n", confusion_matrix(y_test_class, predicted)
    accuracy_class_test = accuracy_score(y_test_class, predicted)
    print "Accuracy on test data for %s : " % which_class, accuracy_class_test, "\n\n"
    return accuracy_class_train, accuracy_class_test


accuracies_elk = estimator(data=X, which_class="Elk", target_vals=is_elk)
accuracies_deer = estimator(data=X, which_class="Deer", target_vals=is_deer)
accuracies_cattle = estimator(data=X, which_class="Cattle", target_vals=is_cattle)

print "Accuracy Score on train data set : %f" % np.mean([accuracies_deer[0], accuracies_cattle[0], accuracies_elk[0]])
print "Accuracy Score on test data set : %f" % np.mean([accuracies_deer[1], accuracies_cattle[1], accuracies_elk[1]])

