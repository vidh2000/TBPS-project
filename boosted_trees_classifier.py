"""
Created on Sun Jan 30 20:14:53 2022

@author: Vid

Extreme Gradient Boosting Classifier with
Scikit-Learn Wrapper interface for XGBoost.

The class contains methods for:
    - training/testing (simultaneously)
    - predicting unknown samples to obtain
      probability arrays for each event.
      This file also outputs provisional signals in a .pkl file
      given some dataset of unknown events
      
      
Have fun!
Will update to use Data Matrix used in XGBoost which is optimised for speed
and memory - probably much faster to train/evaluate on - allows training
on larger datasets. Iterative learning approaches seems to not work, 
even when applied on the whole dataset and repeated over and over again.
NOTES:
Trees are known to be overfitting - include stopping parameters in the future,
cross validation etc.

All datasets are to be situated in data folder in the same directory as this
.py file.

"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from copy import * 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
import xgboost as xgb
from tqdm import tqdm
from scipy.stats import uniform, randint
from sklearn.metrics import *#auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split


#%% Figures sytle
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 16,
          'font.family' : 'lmodern',
          #'text.latex.unicode': True,
          'axes.labelsize':16,
          'legend.fontsize': 11,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'figure.figsize': [7.5,7.5/1.2],
                     }
plt.rcParams.update(params)


#%% Loading files - filenames.
"""
To read the files without modification of the code, they must be all stored
in a folder "data" where data and this python file must be located in the same 
directory.
"""

### Unkown data
# Total dataset - unknown data to analyse and extract signal out of
total_dataset = pd.read_pickle('data/total_dataset.pkl')

### Known - labeled data
# The signal decay, simulated as per the Standard Model
sig = pd.read_pickle('data/signal.pkl')

# B0 ---> J/psi K^*0,     j/psi --> mu mu
jpsi = pd.read_pickle('data/jpsi.pkl')
# B0 ---> psi(2S) K^*0,   psi(2S) --> mu mu
psi2S = pd.read_pickle("data/psi2S.pkl")
# B0 ---> J/psi K^+0 with muon reconstructed as kaon and kaon as muon
jpsi_mu_k_swap = pd.read_pickle("data/jpsi_mu_k_swap.pkl")
# B0 ---> J/psi K^+0 with muon reconstructed as pion and pion as muon
jpsi_mu_pi_swap = pd.read_pickle("data/jpsi_mu_pi_swap.pkl")
# B0 ---> J/psi K^+0 with kaon reconstructed as pion and pion as kaon
k_pi_swap = pd.read_pickle("data/k_pi_swap.pkl")

# B_S^0 --->        \phi mu mu,      \phi --> KK and 1 K reconstructed as pion
phimumu = pd.read_pickle("data/phimumu.pkl")
# Lambda_b^0 --->   pK\mu\mu        with p reconstructed as K and K as pi 
pKmumu_piTok_kTop = pd.read_pickle("data/pKmumu_piTok_kTop.pkl")
# Lambda_b^0 --->   pK\mu\mu        with p reconstructed as pion
pKmumu_piTop = pd.read_pickle("data/pKmumu_piTop.pkl")

# Simulation which is flat in three angular variables and q^2
acceptance = pd.read_pickle("data/acceptance_mc.pkl")

# All files in a list for easier accessibility
datasets = [total_dataset,  #0
            sig,jpsi,psi2S, #1,2,3
            jpsi_mu_k_swap,jpsi_mu_pi_swap,k_pi_swap, #4,5,6
            phimumu,pKmumu_piTok_kTop,pKmumu_piTop, #7,8,9
            acceptance] #10



#%% Data clean-up - always use files in list "datasets"

keys = total_dataset.keys()
datasets = [sig,#0
            jpsi,psi2S, #1,2
            jpsi_mu_k_swap,jpsi_mu_pi_swap,k_pi_swap, #3,4,5
            phimumu,pKmumu_piTok_kTop,pKmumu_piTop, #6,7,8
            acceptance,total_dataset] #9,10


dataset_names = ["Signal",
                 r"J/$\psi$",r"$\psi$(2S)",
                 r"J/$\psi$; $\mu$, $K$ swapped",
                 r"J/$\psi$; $\mu$, $\pi$ swapped",
                 r"J/$\psi$; $K$, $\pi$ swapped",
                 r"$\phi \mu \mu$",
                 r"$p K \mu \mu$; K, $\pi$ swapped",
                 r"$p K \mu \mu$; p, $\pi$ swapped",
                 "flat acceptance",
                 "Total dataset"]

dataset_labels = [i for i in range(len(datasets))]
for i in range(len(datasets)):
    datasets[i] = datasets[i].drop(columns=["year","B0_ID"])
    datasets[i]["file"] = dataset_labels[i]

keys = datasets[i].keys()

#%% DataLoading

# Merge signals dataset to
size_class = 1000
N_classes = 10
merged = datasets[0].iloc[0:1]
#N=int(size_class/6)
#not_sig_length = 0
for i in range(N_classes):
    data = datasets[i]
    #print("i=",i,"len(data)",len(data))
    if len(data)<size_class:
        file = data#.iloc#[0:1]
        #while len(file)<size_class:
        #    file = file.append(data)
        merged= merged.append(file)
        #print(len(file))
    else:
        file = data.iloc[0:size_class]
        merged = merged.append(file)
        #print(len(file))
 
print("Number of entries:",len(merged))


# =============================================================================
# 
# merged = datasets[1].iloc[0:size_class]
# merged = merged.append(datasets[-1].iloc[0:size_class])
# =============================================================================

# Randomly shuffle the data since now the rows in "merged" are ordered 1..1,2..2,...
merged = merged.sample(frac=1)

# Features on which classifier will learn
# i.e everything but the column "file" which is the label)
X = merged[keys[:-1]]
# Label i.e the column "file"
y = merged[keys[-1]]



# Scale the variables. Each column - feature is normalised;
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


def batch_data(X,y,batch_size=None,N_batches=None):
    """
    Separate datasets X and y into batches.
    Either:
        - of size batch_size
        - number of batches N_batches
    By default, choose the size of batches.
    """
    N=len(y)
    # Specifying number of batches
    if N_batches != None:
        batch_size = int(N/N_batches)
        X_batches = []
        y_batches = []
        i=0
        while i<N:
            Xbatch = X[i:i+batch_size]
            ybatch = y[i:i+batch_size]
            X_batches.append(Xbatch)
            y_batches.append(ybatch)
            i += batch_size
        return X_batches, y_batches
    
    # Specifying the size of batches
    else:
        X_batches = []
        y_batches = []
        i=0
        while i<N:
            Xbatch = X[i:i+batch_size]
            ybatch = y[i:i+batch_size]
            X_batches.append(Xbatch)
            y_batches.append(ybatch)
            i += batch_size
        return X_batches, y_batches

def get_batched_traintest_data(X,y,test_size,batch_size=None,N_batches=None,
                               one_testset=False):
    """
    Separate dataset into training and testing data.
    Create equal number of batches for both.
    """
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size)
    # Batch train data
    X_train_batches,y_train_batches = batch_data(X_train,y_train,
                                                 batch_size,N_batches)
    N_btchs = len(X_train_batches)
    
    if one_testset:
        return X_train_batches, X_test, y_train_batches, y_test
    else:
        # Make equal number of batches from testing data
        X_test_batches,y_test_batches = batch_data(X_test,y_test,N_batches=N_btchs)
        
        return X_train_batches, X_test_batches, y_train_batches, y_test_batches

import time 

### Train ###
xgb_model = xgb.XGBClassifier()
model=None


#batch_size = 1000
N_batches=1
Xtrain_bts,X_test,ytrain_bts,y_test= get_batched_traintest_data(X,y,
                                        test_size=0.25,N_batches=N_batches,
                                        #batch_size=batch_size,
                                        one_testset=True)


#%% Extreme Gradient Boosting


class XGBoost_sklearn():
    """
    XGBoost Extreme Gradient booster classifier
    using Scikit-Learn Wrapper interface for XGBoost.
    Easy to use.
    """
    def __init__(self,X,y,test_size,N_batches=1,
                                     batch_size=10000,one_testset=True):
        
        self.clf = xgb.XGBClassifier()
        self.test_size = test_size
        self.Xtrain_bts,self.X_test,self.ytrain_bts,self.y_test=get_batched_traintest_data(X,y,
                                    test_size=test_size,N_batches=N_batches,
                                    batch_size=batch_size,
                                    one_testset=True)
    def train_test(self):
        start = time.time()
        model=None
        model = self.clf.fit(self.Xtrain_bts[0],self.ytrain_bts[0],
                              xgb_model=model)
        end = time.time()
        print("\nExtreme Gradient Boosting Classifier (xgboost-sklearn)")
        print("Trained for:",end-start,"seconds")
        y_pred = self.clf.predict(self.X_test)
        acc =  self.clf.score(self.X_test,self.y_test)
        print("On testing data of size:",len(self.X_test))
        conf_matrix = confusion_matrix(self.y_test,y_pred)
        print("Confusion matrix:\n",conf_matrix)
        obtained_signal_acc = conf_matrix[0][0]/np.sum(conf_matrix,axis=0)[0]
        print("Overall accuracy:",acc)
        print("Obtained signal accuracy:",obtained_signal_acc)
        
    def get_provisional_signals(self,X_unknown_unscaled,
                        N_classes,size_class,plot_probability_distrib=False):
        """
        Stores the provisional signals in a .pkl file
        and return array of provisional signals
        """
        
        # Get probability arrays
        X_unknown = scaler.transform(X_unknown_unscaled) 
        probability_arrays = self.clf.predict_proba(X_unknown)
        
        if plot_probability_distrib:
            # Plot the distribution of probability values for the classified signals
            plt.figure("signal_probability_distribution")
            plt.hist(probability_arrays,bins=30,histtype="step")
            plt.ylabel("N")
            plt.xlabel("Probability of being a signal")
            plt.show()
    
        # Get (provisional) signals from total_dataset.pkl
        sigs_provis=X_unknown_unscaled[0:0]
        sigcounter=0
        for i in tqdm(range(len(probability_arrays))):
            if np.argmax(probability_arrays[i])==0:
                sig = X_unknown_unscaled.iloc[i]
                sigs_provis = sigs_provis.append(sig)
                sigcounter += 1
                
        print(f"{sigcounter} signals detected in total_dataset.pkl")
        print(f"{sigcounter/len(data_unknown)*100}% of unknown data classified as signal")
        
        filename="provisional_signals/XGBClf_signals_"+f"{N_classes}_classes_"+ \
                f"{size_class}_class_size"+".pkl"
                
        store_obtained_sigs = sigs_provis.to_pickle(filename)
        return sigs_provis
    
model = XGBoost_sklearn(X,y,test_size=0.25)
model.train_test() 


# Now let's test the classifier on the unknown unlabeled data from "total_dataset.pkl"
data_unknown = datasets[-1]
X_unknown_unscaled = data_unknown[keys[:-1]]

sigs_provis = model.get_provisional_signals(X_unknown_unscaled,
                                      N_classes,size_class)












