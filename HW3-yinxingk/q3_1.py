'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import random

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''

        labels_list = self.train_labels.tolist()
        distance_list = self.l2_distance(test_point).tolist()
        z = sorted(zip(distance_list, labels_list))
        # ???
        # sorted_test_points = [x for x,_ in z]
        sorted_labels = [y for _,y in z]

        labels = sorted_labels[:k]
        count = 0
        ties = []
        for l in set(labels):
            cur_count = labels.count(l)
            if count < cur_count:
                ties = [l]
                count = cur_count
            elif count == cur_count:
                ties.append(l)

        return random.choice(ties)

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    score = 0
    amount = len(eval_data)
    for i in range(amount):
        if(eval_labels[i] == knn.query_knn(eval_data[i], k)):
            score += 1
    return score / amount

def predict(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    amount = len(eval_data)
    prediction = []
    for i in range(amount):
        prediction.append(knn.query_knn(eval_data[i], k))
    return np.array(prediction)

def labels_to_one_hot(labels):
    targets = []
    for lbl in labels:
        a = [0.]*10
        a[int(lbl)] = 1.
        targets.append(a)
    return np.array(targets)

def plot_roc(y_test, y_score):
    n_classes = 10
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes

    lw = 2
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC graph of kNN')
    plt.legend(loc="lower right")
    plt.show()

def cross_validation(train_data, train_labels, k):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    # Loop over folds
    # Evaluate k-NN
    # ...
    # print("k: ", k)
    cur_accuracy = []
    sum = 0
    num = 0
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(train_data, train_labels):
        tr_data = np.array([train_data[i] for i in train_index])
        tr_label = np.array([train_labels[i] for i in train_index])
        te_data = np.array([train_data[i] for i in test_index])
        te_label = np.array([train_labels[i] for i in test_index])
        knn = KNearestNeighbor(tr_data, tr_label)
        accuracy = classification_accuracy(knn, k, te_data, te_label)
        cur_accuracy.append(accuracy)
        sum += accuracy
        num += 1
    print("10 foled average accuracy = " ,sum/num)


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    score = 0
    amount = len(eval_data)
    for i in range(amount):
        if(eval_labels[i] == knn.query_knn(eval_data[i], k)):
            score += 1
    return score / amount
    

def main():
    print("Enter main.")
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    # predicted_label = knn.query_knn(test_data[0], 1)
    print("Test accuracy for k=1")
    print(classification_accuracy(knn, 1, test_data, test_labels))
    print("Train accuracy for k=1")
    print(classification_accuracy(knn, 1, train_data, train_labels))
    print("Test accuracy for k=15")
    print(classification_accuracy(knn, 15, test_data, test_labels))
    print("Train accuracy for k=15")
    print(classification_accuracy(knn, 15, train_data, train_labels))
    for k in range(1, 16):
        print("k=", k)
        print("Test accuracy:", classification_accuracy(knn, k, test_data, test_labels))
        print("Train accuracy:", classification_accuracy(knn, k, train_data, train_labels))
        cross_validation(train_data, train_labels, k)

    prediction = predict(knn, 3, test_data, test_labels)
    prediction_proba = labels_to_one_hot(prediction)
    y_test = labels_to_one_hot(test_labels)

    plot_roc(y_test, prediction_proba)

    print("confusion matrix: ")
    print(confusion_matrix(test_labels, prediction))
    print("Precision: ", precision_score(test_labels, prediction, average='macro'))
    print("Recall: ", recall_score(test_labels, prediction, average='macro'))


if __name__ == '__main__':
    main()
