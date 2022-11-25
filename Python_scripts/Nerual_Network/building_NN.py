import numpy as np

class Build:
    def __init__(self, data):
        self.data = data
        self.n_hidden_neurons = 50 #Number of neurons in hidden layer
        self.n_categories = 10 #number of categories as the output (numb. 0 - 9)

        #Defining weights and biases in hidden layer
        self.hidden_weights = np.random.randn(len(self.data[1]), self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        #Defining weights and biases in output layer
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def feed_forward(self):
        #Weighted sum of inputs and the activation to the hidden layer
        self.z_h = np.matmul(self.data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.sigmoid(self.z_h)

        #The same but now in the output layer
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        #z_o shape = (1797, 10)
        #We use softmax output
        exp_term = np.exp(self.z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        predict = np.argmax(probabilities, axis=1)

        return self.a_h, probabilities, predict

    def to_categorical_numpy(integer_vector):
        n_inputs = len(integer_vector)
        n_categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros([n_inputs, n_categories])
        onehot_vector[range(n_inputs), integer_vector] = 1

        return onehot_vector

    def backpropagation(self, X, y):
        a_h, probabilities, predict = self.feed_forward()
        print(predict)

        #Calculating the error in the output layer
        #error_output = probabilities - y
        #Calculating the error in the hidde layer
        #error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)

        #print(error_output)
