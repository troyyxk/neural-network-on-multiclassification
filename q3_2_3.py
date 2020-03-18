import data
import numpy as np
# Import pyplot - plt.imshow is useful!
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt

def main():
    print("Enter main.")
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    clf = AdaBoostClassifier(n_estimators=1000, random_state=0, learning_rate=0.9, algorithm='SAMME')
    clf.fit(train_data, train_labels)
    prediction = clf.predict(test_data)
    print("Accuracy: ", clf.score(test_data, test_labels))
    print("confusion matrix: ")
    print(confusion_matrix(test_labels, prediction))
    print("Precision: ", precision_score(test_labels, prediction, average='macro'))
    print("Recall: ", recall_score(test_labels, prediction, average='macro'))


if __name__ == '__main__':
    main()