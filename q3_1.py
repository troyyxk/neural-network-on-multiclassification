'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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
        for l in set(labels):
            cur_count = labels.count(l)
            if count < cur_count:
                count = cur_count
                digit = l
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    all_accuracy = []
    for k in k_range:
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
        print("k= ", k, ", accuracy = " ,sum/num)
        all_accuracy.append(cur_accuracy)
        # print("cur_accuracy: ", cur_accuracy)
    # print("all_accuracy: ", all_accuracy)
    print 
        


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
    cross_validation(train_data, train_labels)
    cross_validation(test_data, test_labels)


if __name__ == '__main__':
    main()