import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(0)

# generating a synthetic dataset
X = np.random.randn(1000, 2)  # 1000 samples and 2 features
Y = np.random.randint(0, 5, size=(1000, 1))  # five classes 


class NeuralNetwork(object):
    def __init__(self):
        inputLayerNeurons = 2
        hiddenLayerNeurons = 10
        outLayerNeurons = 5  # five output neurons for multi-class classification

        self.learning_rate = 0.2
        self.W_HI = np.random.randn(inputLayerNeurons, hiddenLayerNeurons)
        self.W_OH = np.random.randn(hiddenLayerNeurons, outLayerNeurons)

    def sigmoid(self, x, der=False):
        if der == True:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def feedForward(self, X):
        hidden_input = np.dot(X, self.W_HI)
        self.hidden_output = self.sigmoid(hidden_input)

        output_input = np.dot(self.hidden_output, self.W_OH)
        pred = self.softmax(output_input)  # softmax activation for multi-class classification
        return pred

    def backPropagation(self, X, Y, pred):
        m = len(Y)
        output_error = pred - Y

        output_delta = self.learning_rate * output_error

        hidden_error = output_delta.dot(self.W_OH.T)
        hidden_delta = self.learning_rate * hidden_error * self.sigmoid(self.hidden_output, der=True)

        self.W_HI -= X.T.dot(hidden_delta) / m
        self.W_OH -= self.hidden_output.T.dot(output_delta) / m

    def train(self, X, Y):
        output = self.feedForward(X)
        self.backPropagation(X, Y, output)


# split the dataset into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# initialize the neural network
NN = NeuralNetwork()

# training the neural network
err = []
epochs = 10000
for i in range(epochs):
    NN.train(X_train, Y_train)
    err.append(np.mean(np.square(Y_train - NN.feedForward(X_train))))

plt.plot(err)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss')
plt.show()

# evaluate the performance of the trained model using the training data
pred_train = NN.feedForward(X_train)
predicted_train_classes = np.argmax(pred_train, axis=1)
accuracy_train = np.mean(predicted_train_classes == Y_train.flatten())

print("Training Accuracy:", accuracy_train)

# evaluate the performance of the trained model using the testing data
pred_test = NN.feedForward(X_test)
predicted_test_classes = np.argmax(pred_test, axis=1)
accuracy_test = np.mean(predicted_test_classes == Y_test.flatten())

print("Testing Accuracy:", accuracy_test)

# visualize the confusion matrix
conf_matrix = confusion_matrix(Y_test.flatten(), predicted_test_classes)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# print classification report
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
print(classification_report(Y_test.flatten(), predicted_test_classes, target_names=class_names))

# example of comparing models with different configurations
NN1 = NeuralNetwork()
NN2 = NeuralNetwork()
NN3 = NeuralNetwork()

# train each model
for i, NN_model in enumerate([NN1, NN2, NN3]):
    for epoch in range(epochs):
        NN_model.train(X_train, Y_train)

    # evaluate on testing data
    pred_test = NN_model.feedForward(X_test)
    predicted_test_classes = np.argmax(pred_test, axis=1)
    accuracy_test = np.mean(predicted_test_classes == Y_test.flatten())

    print(f"Model {i+1} Testing Accuracy:", accuracy_test)
