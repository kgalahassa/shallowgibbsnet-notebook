# shallowgibbsnet-notebook
This notebook is set to describe how to run the Shallow Gibbs Neural Network Model

Import our dataset: slump dataset (here as our e.g.)


```python
import pandas as pd
import random
import numpy as np
from numpy import linalg as LA


##################################################################################### COPULE DATA ###################################################################################

from scipy.io import arff
data = arff.loadarff('slump.arff')
df = pd.DataFrame(data[0])

covariables = df.iloc[:,0:7].values
response = df.iloc[:,7:10].values
positions = np.arange(103)

from sklearn.model_selection import train_test_split

covariables_train, covariables_test, response_train, response_test,positions_train,positions_test = train_test_split(covariables, response,positions, test_size=0.33, random_state=42)

####################################################################################################################################################################################
####################################################################################################################################################################################

#Pour le training, build training data
xtrain,ytrain = covariables_train, response_train
#corrections

np.where(np.isnan(ytrain), ma.array(ytrain, mask=np.isnan(ytrain)).mean(axis=0), ytrain)


#Pour tester build test data
xtest,ytest = covariables_test, response_test 

#corrections

np.where(np.isnan(ytest), ma.array(ytest, mask=np.isnan(ytest)).mean(axis=0), ytest) 
#####################################################################
```


```python
#######################################################################################################################################
#an integer for the number of neurons on the hidden layer
my_l_1 = NHLayer = 2


#n integer for the number of epochs training times
epoch_times =  args.epoch_times

#a float for the learning rate of parameters
learning_rate = args.learning_rate

#an integer for the proportion of batch training data
batch_psize = args.batch_psize

######################################################################################################################################

#an integer for the Number of times we simulate \lambda_w to estimate the Expected-loglikelihood Score
Number_EpLogLike = args.Number_EpLogLike 


#an integer for the Number of times We simulate  W and b to be used for sampling the predicted test data from the model
Simulate_W_b_Pred = 5

#an integer for the Number of times we simulate Ytest given [each] W and b
Pred_Simulate_Ytest = 5

#an integer for the Number of times We simulate  W and b to be used to estimate the probability of acceptation each partition
Simulate_Proba_Partition = 5 

######################################################################################################################################
#covtraining = args.covtraining

######################################################################################################################################

#an integer for the number of Double Backpropagation Correction
DBS = 20

#a float for the DBS learning rate of parameters
dbs_epsilon = 10e-3

#training and testing data

Train_PottsData = xtrain
Test_PottsData = xtest
```

Run an initial Partition first using the pottscompleteshrinkage package


```python
Train_PottsData_demo = xtrain
#Import the Potts Complete Shrinkage module
import pottsshrinkage.completeshrinkage as PCS
#Choose the number of colors
q = 20
#Compute Initial Potts Clusters as a first Random Partition (with Potts Model)
InitialPottsClusters = PCS.InitialPottsConfiguration(Train_PottsData_demo, q, Kernel="Mercer")
#Choose your temperature (T) level
T = 1000

#Set the bandwidth of the model
sigma = 1

#Set the Number of Random_Partitions you want to simulate
Number_of_Random_Partitions = 1

#Set your initial (random) Potts partition as computed above
Initial_Partition = InitialPottsClusters

#Set the Minimum Size desired for each partition generated
MinClusterSize = 5

#Run your Potts Complete Shrinkage Model to simulate the Randomly Shrunk Potts Partitions. Partitions_Sets is a dictionary that can be saved 
#with pickle package.
Partitions_Sets,Spin_Configuration_Sets = PCS.Potts_Random_Partition (Train_PottsData_demo, T, sigma, Number_of_Random_Partitions, MinClusterSize, Initial_Partition,  Kernel="Mercer")

```

Save the initial Partition from Partitions_Sets


```python
import pickle
output = open('Initial_Partitions_constraints_1_Sets.pkl.pkl', 'wb')
pickle.dump(Partitions_Sets, output)
output.close()
```

Run the model in a finite loop for a maximum number of desired partitions such that we stop the loop when completed.

The model has to reject the partition that are generated but not accepted by an acceptation ratio

You need to install the package shallowgibbsnet (pip install shallowgibbsnet), then import the desired structure:
    
    1- BetweenLayerSparsed (import shallowgibbs.BetweenLayerSparsed.sgnn as ShallowGibbs)
    2- CompoundSymmetry (import shallowgibbs.CompoundSymmetry.sgnn as ShallowGibbs)
    3- FullySparsed (import shallowgibbs.FullySparsed.sgnn as ShallowGibbs)
    4- FullyConnected (import shallowgibbs.FullyConnected.sgnn as ShallowGibbs)
    5- SparsedCompoundSymmetry (import shallowgibbs.SparsedCompoundSymmetry.sgnn as ShallowGibbs)


```python
#For example: to import the BetweenLayerSparsed model we have:
import shallowgibbs.BetweenLayerSparsed.sgnn as ShallowGibbs
```


```python
Maximum_number_of_partitions = 10

for i in range(Maximum_number_of_partitions):

    
    ShallowGibbs.ShallowGibbsProcedure(New_partition, xtrain, ytrain, Partitions_Sets,partition_position, xtest,ytest,my_l_1,epoch_times, learning_rate,batch_psize,Number_EpLogLike, Simulate_W_b_Pred, Pred_Simulate_Ytest,Simulate_Proba_Partition, DBS, dbs_epsilon)
    
    acceptation_ratio = ...[]...
    
```



```python
Run the model in a finite loop for a maximum number of desired partitions such that we stop the loop when completed.

The model has to reject the partition that are generated but not accepted by an acceptation ratio
```


```python
Maximum_number_of_partitions = 10

for i in range(Maximum_number_of_partitions):

    
    ShallowGibbsProcedure(New_partition, xtrain, ytrain, Partitions_Sets,partition_position, xtest,ytest,my_l_1,epoch_times, learning_rate,batch_psize,Number_EpLogLike, Simulate_W_b_Pred, Pred_Simulate_Ytest,Simulate_Proba_Partition, DBS, dbs_epsilon)
    
    acceptation_ratio = ...[]...
    
```

