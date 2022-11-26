import numpy as np
import matplotlib.pyplot as plt

from pre_process_data import PreProcessedData
from building_NN import Build

data_inst = PreProcessedData()
data, target = data_inst.data()
data_train, data_test, target_train, target_test = data_inst.train_test(0.8)


inst = Build(data, target)
hw, hb, ow, ob = inst.train()
probabilities = inst.feed_forward(hw, hb, ow, ob, data)

prediction = inst.results(probabilities)

for i in range(1000):
    print(target[i], prediction[i])
