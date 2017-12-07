def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    ### create classifier
    clf = GaussianNB()
    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)
    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = count2(pred, labels_test)
    return accuracy

def count(pred, label_test):
    count = 0.0
    for i in range(0, len(pred)):
        if pred[i] == labels_test[i]:
            count += 1
    accuracy = count / len(pred)
    return accuracy


def count2(pred, labels_test):
    from sklearn.metrics import accuracy_score
    return accuracy_score(labels_test, y_pred = pred)