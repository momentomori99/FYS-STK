import numpy as np
from sklearn.metrics import accuracy_score

class Build:
    def __init__(self, data, target):
        self.data = data
        self.n_hidden_neurons = 50 #Number of neurons in hidden layer
        self.n_categories = 10 #number of categories as the output (numb. 0 - 9)
        self.n_inputs = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self.onehot_vector = self.onehot(target)
        self.epochs = 10
        self.batch_size = 100
        self.iterations = self.n_inputs // self.batch_size
        self.eta = 0.01
        self.lmbd = 0.01


        #Defining weights and biases in hidden layer
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        #Defining weights and biases in output layer
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def predict(self):
        a_h, probabilities = self.feed_forward()
        return np.argmax(probabilities, axis=1)

    def results(self, probabilities):
        return np.argmax(probabilities, axis=1)

    def onehot(self, integer_vector):
        n_inputs = len(integer_vector)
        n_categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros([n_inputs, n_categories])
        onehot_vector[range(n_inputs), integer_vector] = 1

        return onehot_vector

    def feed_forward(self, hidden_weights, hidden_bias, output_weights, output_bias, X):
        #Weighted sum of inputs and the activation to the hidden layer
        self.z_h = np.matmul(X, hidden_weights) + hidden_bias
        self.a_h = self.sigmoid(self.z_h)

        #The same but now in the output layer
        self.z_o = np.matmul(self.a_h, output_weights) + output_bias

        #We use softmax output
        exp_term = np.exp(self.z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        #We wish to have an array in same shape as target array, this predict
        #array contains number which our FFNN predicted with highes probabilities.


        return probabilities

    def backpropagation(self, X, Y):
        probabilities = self.feed_forward(self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias, X)
        #Calculating the error in the output layer
        error_output = probabilities - Y
        #Calculating the error in the hidde layer
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)


        #gradient for the output layer:
        output_weights_gradient = np.matmul(self.a_h.T, error_output)
        output_bias_gradient = np.sum(error_output, axis=0)

        # gradient for the hidden layer
        hidden_weights_gradient = np.matmul(X.T, error_hidden)
        hidden_bias_gradient = np.sum(error_hidden, axis=0)


        return hidden_weights_gradient, hidden_bias_gradient, output_weights_gradient, output_bias_gradient

    def train(self):
        data_indices = np.arange(self.n_inputs)
        for i in range(self.epochs):
            for j in range(self.iterations):
                #pick random datapoints:
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )
                X = self.data[chosen_datapoints]
                Y = self.onehot_vector[chosen_datapoints]

                dWh, dBh, dWo, dBo = self.backpropagation(X, Y)

                if self.lmbd > 0.0:
                    dWo += self.lmbd * self.output_weights
                    dWh += self.lmbd * self.hidden_weights
                self.output_weights -= self.eta * dWo
                self.output_bias -= self.eta * dBo
                self.hidden_weights -= self.eta * dWh
                self.hidden_bias -= self.eta * dBh

        return self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias
