import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download

def download_data():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_dir = "./data"
    download.maybe_download_and_extract(url, download_dir)

# Class to initialize and apply K-nearest neighbour classifier
class KNearestNeighbor(object):
    def __init__(self):
        pass

    # Method to initialize classifier with training data
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # Method to predict labels of test examples using 'compute_distances' and 'predict_labels' methods.
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)

    # Method to compute Euclidean distances from each text example to every training example  
    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # Compute distances from each test example (in argument 'X' of this method) to every training example and store distances in
        # dists variable given above. For each row, i, dist[i] should contain distances between test example i and every training example.
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1))
        return dists

    # Method to predict labels of test examples using chosen value of k given Euclidean distances obtained from 'compute_distances' method.
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        # Given dists computed using 'compute_distances' method above, obtain k closest distances to training examples for each test example
        # dists[i]. Use k closest distances obtained to predict label of each dists[i]. Label of each dists[i] should be stored in y_pred[i].
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

def visualize_data(X_train, y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

if __name__ == "__main__":
    # download_data()
    cifar10_dir = 'D:\\AIUB\\Academics\\Semester 10\\ML\\Mid Assignment\\Assignment 1\\KNN-HW\\cifar-10-batches-py'
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)

    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    visualize_data(X_train, y_train)
    num_training = 8000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 2000
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print(X_train.shape, X_test.shape)

    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=5)

    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct with k=5 => accuracy: %f' % (num_correct, num_test, accuracy))

    num_folds = 5
    k_choices = [1, 3, 5, 8, 10]
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    k_to_accuracies = {}

    for k in k_choices:
        k_to_accuracies[k] = []
        for i in range(num_folds):
            print("debug")
            X_train_cv = np.vstack(X_train_folds[:i] + X_train_folds[i + 1:])
            y_train_cv = np.hstack(y_train_folds[:i] + y_train_folds[i + 1:])
            X_val_cv = X_train_folds[i]
            y_val_cv = y_train_folds[i]
            classifier_cv = KNearestNeighbor()
            classifier_cv.train(X_train_cv, y_train_cv)
            dists = classifier_cv.compute_distances(X_val_cv)
            y_val_pred = classifier_cv.predict_labels(dists, k=k)
            num_correct = np.sum(y_val_pred == y_val_cv)
            accuracy = float(num_correct) / len(y_val_cv)
            k_to_accuracies[k].append(accuracy)

    print("Printing our 5-fold accuracies for varying values of k:")
    print("working")
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    for k in sorted(k_to_accuracies):
        print('k = %d, avg. accuracy = %f' % (k, sum(k_to_accuracies[k]) / num_folds))

    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.savefig('cross-validation_accuracy.jpg')

    best_k = max(k_to_accuracies, key=lambda k: np.mean(k_to_accuracies[k]))

    classifier_best_k = KNearestNeighbor()
    classifier_best_k.train(X_train, y_train)
    y_test_pred_best_k = classifier_best_k.predict(X_test, k=best_k)

    num_correct = np.sum(y_test_pred_best_k == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct on test data => accuracy: %f' % (num_correct, num_test, accuracy))