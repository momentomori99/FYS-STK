import numpy as np
import matplotlib.pyplot as plt

from pre_process_data import PreProcessedData
from building_NN import Build

data_inst = PreProcessedData()
data, target = data_inst.data()
data_train, data_test, target_train, target_test = data_inst.train_test(0.8)


inst = Build(data, target)
inst.backpropagation()
# build_inst.backpropagation(data, target)
# build_inst.gradient_descent(data, target)
