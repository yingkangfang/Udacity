# -*- coding:utf-8 -*-
#!/usr/bin/python

import sys
import pickle
import numpy
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import sklearn.metrics
import time


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees','to_messages',
                  'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# 数据集的探索：包括数据点总数，poi占比，特征的数量，那些特征值有很多缺省值
print "the number of people in dataset: %d" % len(data_dict)
print "the number of features: %d" % len(data_dict.values()[0].keys())
poi_count = 0
for people in data_dict.values():
    if people['poi']:
        poi_count += 1
print "the number of POI: %d, the proportion in the total people: %2d%%" %(poi_count, float(poi_count)/len(data_dict)*100)
all_features = data_dict.values()[0].keys()
print "the features in dataset: %s" % all_features
print

# 初始化缺省特征值的计数
miss_features = {}
for feature in all_features:
    miss_features[feature] = 0
# 统计缺省值
for people in data_dict.values():
    for feature in all_features:
        if people[feature] == 'NaN':
            miss_features[feature] += 1
print "miss_features:", miss_features
for key, value in miss_features.items():
    if value >= 100:
        print "%s : %d " % (key, value)



### Task 2: Remove outliers
import matplotlib.pyplot as plt
data = featureFormat(data_dict, features_list, remove_NaN=True, remove_all_zeroes=True )

# 通过散点图来观察是否有异常值
def check_outlier(data, feature_a, feature_b, features_list):
    a = features_list.index(feature_a)
    b = features_list.index(feature_b)
    for point in data:
        x = point[a]
        y = point[b]
        if point[0]:
            plt.scatter(x, y, color="r")
        else:
            plt.scatter(x, y, color="b")
    plt.xlabel(feature_a)
    plt.ylabel(feature_b)
    plt.show()
check_outlier(data, 'salary', 'total_stock_value', features_list)

# 去除异常值
data_dict.pop("TOTAL")
data = featureFormat(data_dict, features_list, remove_NaN=True, remove_all_zeroes=True)
# check_outlier(data, 'salary', 'total_stock_value', features_list)
# check_outlier(data, 'from_messages', 'to_messages', features_list)
"""
for key, value in data_dict.items():
    if value['from_messages'] >= 14000 and value['from_messages'] != 'NaN':
        print "from_message >= 14000"
        print key, value['from_messages']
    if value['to_messages'] >= 10000 and value['to_messages'] != 'NaN':
        print "to_messages >= 10000"
        print key, value['to_messages']
"""
# check_outlier(data, 'from_this_person_to_poi','from_poi_to_this_person', features_list)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# 创建新的特征值
for value in my_dataset.values():
    from_this_person_to_poi = float(value['from_this_person_to_poi'])
    from_this_person_to_all = float(value['from_messages'])
    if from_this_person_to_all > 0:
        value['percent_to_poi'] = from_this_person_to_poi/from_this_person_to_all
    else:
        value['percent_to_poi'] = 0

    from_poi_to_this_person = float(value['from_poi_to_this_person'])
    from_all_to_this_person = float(value['to_messages'])
    if from_all_to_this_person > 0:
        value['percent_from_poi'] = from_poi_to_this_person / from_all_to_this_person
    else:
        value['percent_from_poi'] = 0

print "the new all features:", my_dataset.values()[0].keys()
print 'percent_from_poi' in my_dataset.values()[0].keys()

features_list.extend(['percent_from_poi', 'percent_to_poi'])

### Extract features and labels from dataset for local testing
print "the features in features_list:", features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
check_outlier(data, 'percent_from_poi', 'percent_to_poi', features_list)

labels, features = targetFeatureSplit(data)

# 利用selectfrommodel选择特征影响程度高于0.05的特征值
def selectfeatures(features, labels, features_list):
    clf = tree.DecisionTreeClassifier()
    sfm = SelectFromModel(clf, threshold=0.05)
    sfm.fit(features, labels)
    selected_features = ['poi']
    for index, value in enumerate(sfm.get_support().flat):
        if value:
            selected_features.append(features_list[index + 1])
    n_features = sfm.transform(features).shape[1]
    print "the number of selected features:", n_features
    return selected_features

selected_features = selectfeatures(features, labels, features_list)
print "the final selected features are:", selected_features
print

data = featureFormat(my_dataset, selected_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

clf = tree.DecisionTreeClassifier()
# clf = svm.SVC()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# 参数调优
parameters = {'splitter':('best','random'),
              'max_depth':[2,3,4,5],
              'min_samples_split':[2,3,4,5]}
clf_parameters = GridSearchCV(clf, parameters, scoring='f1')
clf_parameters.fit(features, labels)
print clf_parameters.best_params_
clf = clf_parameters.best_estimator_

# 评估算法
def evaluation(labels_test, pred):
    print "accuracy score:", sklearn.metrics.accuracy_score(labels_test, pred)
    print "precision score:", sklearn.metrics.precision_score(labels_test, pred)
    print "recall score:", sklearn.metrics.recall_score(labels_test, pred)
    print

# Kflod 验证算法性能
from sklearn.model_selection import KFold
kf = KFold(n_splits = 10)
for train_index, test_index in kf.split(features):
    features_train = [features[i] for i in train_index]
    features_test = [features[i] for i in test_index]
    labels_train = [labels[i] for i in train_index]
    labels_test = [labels[i] for i in test_index]
    t0 = time.time()
    clf.fit(features_train, labels_train)
    print "training time: %f s" % round(time.time() - t0, 3)
    pred = clf.predict(features_test)
    evaluation(labels_test, pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, selected_features)