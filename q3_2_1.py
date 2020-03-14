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
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 10)

        self.hidden = nn.Linear(784, 256)

    def forward(self, x):
        x= torch.relu(self.fc1(x))
        x= torch.relu(self.fc2(x))
        x= torch.relu(self.fc3(x))
        x= self.fc4(x)
        
        return F.softmax(x, dim=1)


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


def labels_to_one_hot(labels):
    tensors = []
    for lbl in labels:
        a = [0.]*10
        a[int(lbl)] = 1.
        tensors.append(a)
    return torch.tensor(tensors)

def classification_accuracy(net, test_data_tensor, test_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    score = 0
    amount = len(test_labels)
    for i in range(amount):
        output = net(test_data_tensor[i].view(-1, 64).float())
        for idx, i in enumerate(output):
            if torch.argmax(i) == test_labels[idx]:
                score += 1
    return score / amount


def main():
    print("Enter main.")
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # convert to tensor
    train_data_tensor = torch.from_numpy(train_data)
    train_labels_tensor = labels_to_one_hot(train_labels)
    test_data_tensor = torch.from_numpy(test_data)

    net = Net()
    net = net.float()
    optimizer = optim.Adam(net.parameters(), lr = 0.01)
#     optimizer.zero_grad()

#     criterion = nn.CrossEntropyLoss()

    # train 3 times
    for _ in range(3):
        for i in range(len(train_data_tensor)):
            net.zero_grad()
            output = net(train_data_tensor[i].view(-1, 64).float())
            loss = F.mse_loss(output, train_labels_tensor[i])
#             print(output)
#             print(train_labels_tensor[i])
#             loss = criterion(output, train_labels_tensor[i])

            loss.backward()
            optimizer.step()

    print(classification_accuracy(net, train_data_tensor, train_labels))
    print(classification_accuracy(net, test_data_tensor, test_labels))
            
    
            
            
# #     print(train_data_tensor.float())
#     outputs = net(train_data_tensor.float())
#     print(outputs)
#     loss = F.mse_loss(outputs, train_labels_tensor)
#     loss.backward()
#     optimizer.step()
    
    


if __name__ == '__main__':
    main()