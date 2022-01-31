# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:22:17 2022

@author: Vid
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from copy import * 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier

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
#%% q^2 distributions

# =============================================================================
# # Plot distributions of q2 for each file type. As seen, signal looks a lot like
# # combinatorial background. Play around.
# plt.figure("q^2_distributions")
# for i in range(11):
#     name = dataset_names[i]    
#     plt.hist(datasets[i]["q2"],bins="auto",histtype="step",label=name)
# 
# plt.legend()
# plt.ylabel("N")
# plt.xlabel(r"$q^2$")
# plt.show()
# 
# =============================================================================

plt.figure("flat_q2")
plt.hist(acceptance["q2"],bins="auto",histtype="step")
plt.xlabel("q2")

plt.figure("flat_phi")
plt.hist(acceptance["phi"],bins="auto",histtype="step")
plt.xlabel("phi")

plt.figure("flat_costheta_l")
plt.hist(acceptance["costhetal"],bins="auto",histtype="step")
plt.xlabel("costhetal")

plt.figure("flat_costheta_k")
plt.hist(acceptance["costhetak"],bins="auto",histtype="step")
plt.xlabel("costhetak")

plt.show()


#%% Random Forest classifier Class

class Random_Forest():
	"""
	Random Forest machine learning algorithm.
	"""
	def __init__(self, X_train, y_train):#, parameters):
		self.X_train 		= X_train
		self.y_train 		= y_train
		
		# Initialisation of the class trains the model
		print("\nRF - Random forest classifier (sklearn)")
		

		### Classifier - no hyperparameter tuning ###
		self.clf = RandomForestClassifier()
		self.clf.fit(self.X_train, self.y_train)
		

	def get_confusion_matrix(self, X_test, y_test):
		"""
		Prints useful information and returns the confusion matrix from
        being applied on the test data.
		"""
		self.X_test 	= X_test
		self.y_test 	= y_test

		predict_rf = self.clf.predict(self.X_test)
		acc_rf =  self.clf.score(self.X_test,self.y_test)

		confusionmatrix = confusion_matrix(y_true = self.y_test, y_pred= predict_rf)

		# Print accuracy, dataset and confusion matrix(y_true, y_predict)
		print("Accuracy (non-optimized):", acc_rf)
		print("Confusion matrix:\n", confusionmatrix,"\n")


		return confusionmatrix 


	def get_probability_array(self, X_val):
		"""
		Returns probabilities array for unknown data
		"""
		self.X_val = X_val
		probabilities = self.clf.predict_proba(self.X_val)

		return probabilities

#%% Training on 2 classes  (sig/not sig)
# Merge signals dataset to
size_class = 10000
merged = datasets[0].iloc[0:1]
#N=int(size_class/6)
#not_sig_length = 0
for i in range(9):
    data = datasets[i]
    #print("i=",i,"len(data)",len(data))
    if len(data)<size_class:
        file = data.iloc[0:1]
        while len(file)<size_class:
            file = file.append(data)
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

# Both X and y are ordered the same. Essentially, I just "separate" the last column
# from the rest of the dataset which remains the same

# X === i.e 9775 events (rows) with 79 features each
print(X.shape)
# y == i.e labels for corresponding 9775 events (rows)
print(y.shape)

# Scale the variables. Each column - feature is normalised;
# each feature is centered at the mean and divided by the standard diviation
# which makes it easier for classifier to learn on it.
# Since for every feature each value is now 
# "some distance away from that feature's mean"

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Separate into testing and training datasets (X and y are "cut at some row)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

### Train ###

# Initialise the classifying algorithm - random forest here
RF = Random_Forest(X_train,y_train)
print("Training finished")

# Get confusion matrix from test data
confusion_matrx = RF.get_confusion_matrix(X_test,y_test)

# =============================================================================
# 
# # Now let's test the classifier on the unknown unlabeled data from "total_dataset.pkl"
# data_unknown = datasets[0]
# 
# # We'll feed it all columns except the one with label i.e column "file", since
# # "file"==0 for all the data ini total_dataset.pkl
# X_unknown = scaler.transform(data_unknown[keys[:-1]])
# 
# # Get probability arrays i.e for 10 features and 1000 rows, you get 
# # an array of shape (1000,10), where array[:][0] is the probability of 
# # event being the signal
# probs = RF.get_probability_array(X_unknown)
# 
# # Al events where max(probability_array_row) == 0th entry
# # i.e argmax(probability_array_row)=0 are classified as the signal
# 
# signals = [arr[0] for arr in probs if np.argmax(arr)==0]
# 
# # Plot the distribution of probability values for the signal
# # (for all rows/events which were classified as the signal)
# plt.figure("signal_probability_distribution")
# plt.hist(signals,bins=30,histtype="step")
# plt.ylabel("Density")
# plt.xlabel("Probability")
# plt.show()
# 
# 
# =============================================================================














