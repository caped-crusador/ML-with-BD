import pandas as pd
import numpy as np
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_rel


# loading the data from a csv file
raw_data = pd.read_csv("animals.csv")


# see first five rows of data
raw_data.head(n=5)

# Slicing the data into train and test sets
# training_data = raw_data.iloc[0:4000, :]
# testing_data = raw_data.iloc[4001:5134, :]

# summary Statistics of all the attributes in the data\z
raw_data.describe()

# removing attributes that will have very little or no effect on classifier

del raw_data["speed_min"]
del raw_data["acc_mean"]
del raw_data["acc_std"]

# binarification of "class" attribute into three dummmy variables
is_deer = [1 if raw_data["class"][i] == "DEER" else 0 for i in range(len(raw_data["class"]))]
is_cattle = [1 if raw_data["class"][i] == "CATTLE" else 0 for i in range(len(raw_data["class"]))]
is_elk = [1 if raw_data["class"][i] == "ELK" else 0 for i in range(len(raw_data["class"]))]


# conversion of data into NumPy array
data = raw_data.values
X = np.array(data[:, :22])

# creating integer valued classes DEER = 0, CATTLE = 2, ELK =3
# y = []
# for i in raw_data['class']:
#     if i == "DEER":
#         y.append(0)
#     elif i == "CATTLE":
#         y.append(1)
#     else:
#         y.append(2)
#
# print y

kfold = KFold(n_splits=10, random_state=10, shuffle=True)


def estimator_RF(data, which_class, target_vals, kfold):
    print "----------------------------Random Forest---------------------------------"
    # print "----------------------------For %s or not %s------------------------------" % (which_class, which_class)
    # Splitting the data into train and test data sets
    X_train, X_test, y_train_class, y_test_class = train_test_split(data, target_vals, random_state=10)

    # Training the classifier
    RFclf = RandomForestClassifier()
    RFclf.fit(X_train, y_train_class)

    # print "Running classifier on train data"
    # predicted_train = RFclf.predict(X_train)
    # print "Confusion Matrix on train data : \n", confusion_matrix(y_train_class, predicted_train)
    # accuracy_class_train = accuracy_score(y_train_class, predicted_train)
    # print "Accuracy for %s on train data is : %f" % (which_class, accuracy_class_train), "\n\n"

    # print "Running classifier on test data"
    # predicted = RFclf.predict(X_test)
    # print "Confusion matrix : \n", confusion_matrix(y_test_class, predicted)
    # accuracy_class_test = accuracy_score(y_test_class, predicted)
    # print "Accuracy on test data for %s : " % which_class, accuracy_class_test, "\n\n"

    scores_RF = cross_val_score(RFclf, X_test, y_test_class, cv=kfold)
    print "10 fold cross_validation scores for %s : \n" % which_class, scores_RF

    return scores_RF


validation_scores_rf_elk = estimator_RF(data=X, which_class="Elk", target_vals=is_elk, kfold=kfold)
validation_scores_rf_deer = estimator_RF(data=X, which_class="Deer", target_vals=is_deer, kfold=kfold)
validation_scores_rf_cattle = estimator_RF(data=X, which_class="Cattle", target_vals=is_cattle, kfold=kfold)
total_CV_accuracies_rf = np.concatenate((validation_scores_rf_deer ,validation_scores_rf_elk ,validation_scores_rf_cattle))

print "Accuracy : %0.2f (+/-%0.2f)" % (total_CV_accuracies_rf.mean(), total_CV_accuracies_rf.std())

# -----------------------------------------------------------------------------------------------------


def estimator_DT(data, which_class, target_vals, kfold):
    print "----------------------------Decision Tree---------------------------------"
    # print "----------------------------For %s or not %s------------------------------" % (which_class, which_class)
    # Splitting the data into train and test data sets
    X_train, X_test, y_train_class, y_test_class = train_test_split(data, target_vals, random_state=10)

    # Training the classifier
    DTclf = DecisionTreeClassifier()
    DTclf.fit(X_train, y_train_class)

    # print "Running classifier on train data"
    # predicted_train = DTclf.predict(X_train)
    # print "Confusion Matrix on train data : \n", confusion_matrix(y_train_class, predicted_train)
    # accuracy_class_train = accuracy_score(y_train_class, predicted_train)
    # print "Accuracy for %s on train data is : %f" % (which_class, accuracy_class_train), "\n\n"
    #
    # print "Running classifier on test data"
    # predicted = DTclf.predict(X_test)
    # print "Confusion matrix : \n", confusion_matrix(y_test_class, predicted)
    # accuracy_class_test = accuracy_score(y_test_class, predicted)
    # print "Accuracy on test data for %s : " % which_class, accuracy_class_test, "\n\n"

    scores_DT = cross_val_score(DTclf, X_test, y_test_class, cv=kfold)
    print "10 fold cross_validation scores for %s : \n" % which_class, scores_DT

    return scores_DT


validation_scores_dt_elk = estimator_DT(data=X, which_class="Elk", target_vals=is_elk, kfold=kfold)
validation_scores_dt_deer = estimator_DT(data=X, which_class="Deer", target_vals=is_deer, kfold=kfold)
validation_scores_dt_cattle = estimator_DT(data=X, which_class="Cattle", target_vals=is_cattle, kfold=kfold)
total_CV_accuracies_dt = np.concatenate((validation_scores_dt_deer, validation_scores_dt_elk,
                                         validation_scores_dt_cattle))

print "\nAccuracy : %0.2f (+/-%0.2f)" % (total_CV_accuracies_dt.mean(), total_CV_accuracies_dt.std())

# ------------------------------------------------------------------------------------------------------------------------------


def estimator_LR(data, which_class, target_vals, kfold):
    print "----------------------------Logistic Regression---------------------------"
    # print "----------------------------For %s or not %s------------------------------" % (which_class, which_class)
    # Splitting the data into train and test data sets
    X_train, X_test, y_train_class, y_test_class = train_test_split(data, target_vals, random_state=10)

    # Training the classifier
    LRclf = LogisticRegression()
    LRclf.fit(X_train, y_train_class)

    # print "Running classifier on train data"
    # predicted_train = LRclf.predict(X_train)
    # print "Confusion Matrix on train data : \n", confusion_matrix(y_train_class, predicted_train)
    # accuracy_class_train = accuracy_score(y_train_class, predicted_train)
    # print "Accuracy for %s on train data is : %f" % (which_class, accuracy_class_train), "\n\n"
    #
    # print "Running classifier on test data"
    # predicted = LRclf.predict(X_test)
    # print "Confusion matrix : \n", confusion_matrix(y_test_class, predicted)
    # accuracy_class_test = accuracy_score(y_test_class, predicted)
    # print "Accuracy on test data for %s : " % which_class, accuracy_class_test, "\n\n"

    scores_LR = cross_val_score(LRclf, X_test, y_test_class, cv=kfold)
    print "10 fold cross_validation scores for %s : \n" % which_class, scores_LR

    return scores_LR


validation_scores_lr_elk = estimator_LR(data=X, which_class="Elk", target_vals=is_elk, kfold=kfold)
validation_scores_lr_deer = estimator_LR(data=X, which_class="Deer", target_vals=is_deer, kfold=kfold)
validation_scores_lr_cattle = estimator_LR(data=X, which_class="Cattle", target_vals=is_cattle, kfold=kfold)
total_CV_accuracies_lr = np.concatenate((validation_scores_lr_deer, validation_scores_lr_elk,
                                         validation_scores_lr_cattle))

print "Accuracy : %0.2f (+/-%0.2f)" % (total_CV_accuracies_lr.mean(), total_CV_accuracies_lr.std())

# -----------------------------------------------------------------------------------------------------


def estimator_NB(data, which_class, target_vals, kfold):
    print "-------------------------Naive Bayes--------------------------------------"
    # print "----------------------------For %s or not %s------------------------------" % (which_class, which_class)
    # Splitting the data into train and test data sets
    X_train, X_test, y_train_class, y_test_class = train_test_split(data, target_vals, random_state=10)

    # Training the classifier
    NBclf = GaussianNB()
    NBclf.fit(X_train, y_train_class)

    # print "Running classifier on train data"
    # predicted_train = NBclf.predict(X_train)
    # print "Confusion Matrix on train data : \n", confusion_matrix(y_train_class, predicted_train)
    # accuracy_class_train = accuracy_score(y_train_class, predicted_train)
    # print "Accuracy for %s on train data is : %f" % (which_class, accuracy_class_train), "\n\n"
    #
    # print "Running classifier on test data"
    # predicted = NBclf.predict(X_test)
    # print "Confusion matrix : \n", confusion_matrix(y_test_class, predicted)
    # accuracy_class_test = accuracy_score(y_test_class, predicted)
    # print "Accuracy on test data for %s : " % which_class, accuracy_class_test, "\n\n"

    scores_NB = cross_val_score(NBclf, X_test, y_test_class, cv=kfold)
    print "10 fold cross_validation scores for %s : \n" % which_class, scores_NB

    return scores_NB


validation_scores_nb_elk = estimator_NB(data=X, which_class="Elk", target_vals=is_elk, kfold=kfold)
validation_scores_nb_deer = estimator_NB(data=X, which_class="Deer", target_vals=is_deer, kfold=kfold)
validation_scores_nb_cattle = estimator_NB(data=X, which_class="Cattle", target_vals=is_cattle, kfold=kfold)
total_CV_accuracies_nb = np.concatenate((validation_scores_nb_deer, validation_scores_nb_elk,
                                         validation_scores_nb_cattle))

print "Accuracy : %0.2f (+/-%0.2f)" % (total_CV_accuracies_nb.mean(), total_CV_accuracies_nb.std())

# ----------------------------------------------------------------------------------------------------
print "*****************************************************************************************************"
print "\nStudent\'s t test results :"

print "RF vs DT : ", ttest_rel(total_CV_accuracies_rf, total_CV_accuracies_dt)
print "RF vs LR : ", ttest_rel(total_CV_accuracies_rf, total_CV_accuracies_lr)
print "RF vs NB : ", ttest_rel(total_CV_accuracies_rf, total_CV_accuracies_nb)
