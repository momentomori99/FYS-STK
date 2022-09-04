import pandas as pd
from IPython.display import display
import numpy as np
from pylab import plt, mpl

"""
Used in making Data Frame tables. Here we also present how to
visualise frame table data.
"""


#define the data
data = {'First Name': ["Frodo", "Bilbo", "Aragorn II", "Samwise"],
        'Last Name': ["Baggins", "Baggins","Elessar","Gamgee"],
        'Place of birth': ["Shire", "Shire", "Eriador", "Shire"],
        'Date of Birth T.A.': [2968, 2890, 2931, 2980]
        }

data_pandas = pd.DataFrame(data)
#display(data_pandas)

data_pandas = pd.DataFrame(data,index=['Frodo','Bilbo','Aragorn','Sam']) #We change the index from 0,1,2,3... to what we selves choose
#display(data_pandas)


#display(data_pandas.loc['Aragorn']) #Get info about one specific thing


#Appending new data

new_hobbit = {'First Name': ["Peregrin"],
              'Last Name': ["Took"],
              'Place of birth': ["Shire"],
              'Date of Birth T.A.': [2990]
              }
#data_pandas=data_pandas.append(pd.DataFrame(new_hobbit, index=['Pippin']))
#display(data_pandas)



#Setting up matrix to dataframe

np.random.seed(100)
rows = 10
cols = 5
A = np.random.randn(rows,cols)
data_frame = pd.DataFrame(A)
#display(data_frame)
#print(f"mean: {data_frame.mean()}") #takes mean value column by column
#print(f"std: {data_frame.std()}") #takes std value column by column
#display(data_frame**2) #We can also do other operations as well!

data_frame.columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth'] #rename the columns
data_frame.index = np.arange(10)

#print(data_frame["Second"].mean()) #find mean value of "second" column
#print(data_frame.info())
#print(data_frame.describe())

#Plotting and making it cool!
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

data_frame.cumsum().plot(lw=2.0, figsize=(10,6))
plt.show()

data_frame.plot.bar(figsize=(10,6), rot=15)
plt.show()
