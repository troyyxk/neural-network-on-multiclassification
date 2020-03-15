import data
import numpy as np
# Import pyplot - plt.imshow is useful!
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

def main():
    print("Enter main.")
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(train_data, train_labels)
    print(clf.score(test_data, test_labels))

if __name__ == '__main__':
    main()