'''
Question 3.2.1 Pytorch Code

Use Pytorch to buid a neural network.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch import no_grad
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve

from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 10)

    def forward(self, x):
        x= torch.relu(self.fc1(x))
        x= torch.relu(self.fc2(x))
        x= torch.relu(self.fc3(x))
        x= torch.relu(self.fc4(x))
        
        return x
        # return F.softmax(x, dim=1)


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
    plt.title('ROC graph of Neural Network')
    plt.legend(loc="lower right")
    plt.show()


def labels_to_one_hot(labels):
    tensors = []
    for lbl in labels:
        a = [0.]*10
        a[int(lbl)] = 1.
        tensors.append(a)
    return torch.tensor(tensors)

def labels_to_one_hot_np(labels):
    targets = []
    for lbl in labels:
        a = [0.]*10
        a[int(lbl)] = 1.
        targets.append(a)
    return np.array(targets)

def one_hot_to_label(prediction):
    result = []
    for i in range(prediction.shape[0]):
        cur = int(prediction[i].argmax())
        result.append(cur)
    return result

def classification_accuracy(net, data_tensor, labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    score = 0
    amount = len(data_tensor)
    # print(amount)
    prediction = net(data_tensor.float())
    prediction = one_hot_to_label(prediction)
    for i in range(amount):
        if prediction[i] == labels[i]:
            score += 1
    return prediction, score / amount


def main():
    print("Enter main.")
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # convert to tensor
    train_data_tensor = torch.from_numpy(train_data)
    train_labels_tensor = labels_to_one_hot(train_labels)
    test_data_tensor = torch.from_numpy(test_data)
    test_labels_tensor = labels_to_one_hot(test_labels)
    X_test = torch.from_numpy(train_labels)
    y_test = labels_to_one_hot_np(test_labels)
    # print(train_labels_tensor.shape)
    # print(test_labels_tensor.shape)

    net = Net()
    # net.train()
    net = net.float()
#     optimizer = optim.Adam(net.parameters(), lr = 0.15)
#     optimizer = optim.SGD(net.parameters(), lr=0.15,momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=0.15,momentum=0.5)
    loss_func = nn.CrossEntropyLoss()

    # train 10000 times
    for _ in range(10000):
        print(_)
        optimizer.zero_grad()
        output = net(train_data_tensor.float())
        loss = loss_func(output, X_test.long())
        loss.backward()
        optimizer.step()

    with no_grad():
        _, accuracy = classification_accuracy(net, train_data_tensor, train_labels)
        print("Train accuracy: ", accuracy)

    with no_grad():
        prediction, accuracy = classification_accuracy(net, test_data_tensor, test_labels)
        print("Test accuracy: ", accuracy)

    prediction_oh = net(test_data_tensor.float())
    # print("y_test", type(y_test), y_test.shape)
    # print(y_test)
    # print("prediction_oh", type(prediction_oh), prediction_oh.shape)
    # print(prediction_oh)
    plot_roc(y_test, prediction_oh.detach().numpy())

    print("confusion matrix: ")
    print(confusion_matrix(test_labels, prediction))
    print("Precision: ", precision_score(test_labels, prediction, average='macro'))
    print("Recall: ", recall_score(test_labels, prediction, average='macro'))
            

if __name__ == '__main__':
    main()
