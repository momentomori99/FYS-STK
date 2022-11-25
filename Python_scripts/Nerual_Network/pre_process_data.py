import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split


class PreProcessedData:
    def __init__(self):
        #Downloading the dataset:
        self.digits = datasets.load_digits()

        #Defining the data and its labels
        self.inputs = self.digits.images
        self.labels = self.digits.target
        self.n_inputs = len(self.inputs) #number of inputs



        #Flatting out the input matrices:
        self.inputs = self.inputs.reshape(self.n_inputs, -1)
        self.n_features = len(self.inputs[1]) #number of features

        # print("----------------------------------------------------------")
        # print(f"Loading data with {self.n_inputs} number of inputs...")
        # print(f"Number of features for this data: {self.n_features}")
        # print("----------------------------------------------------------")


    def data(self):
        return self.inputs, self.labels

    def train_test(self, train_size):
        X_train, X_test, y_train, y_test = train_test_split(self.inputs, self.labels,
                                        train_size=train_size, test_size=(1-train_size))
        return X_train, X_test, y_train, y_test

    def random_display(self):
        indices = np.arange(self.n_inputs)
        random_indices = np.random.choice(indices, size=5)

        for i, image in enumerate(self.digits.images[random_indices]):
            plt.subplot(1, 5, i+1)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title("Label: %d" % self.digits.target[random_indices[i]])
        plt.show()




if __name__ == '__main__':
    inputs, targets = PreProcessedData().data()
    print(targets)
