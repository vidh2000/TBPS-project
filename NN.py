# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:50:54 2022

@author: Vid
"""
import torch
import time
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

#%% Loading files - filenames

# Total dataset - data to analyse
total_dataset = pd.read_pickle('data/total_dataset.pkl')

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



#%% Data clean-up -always use files "datasets"

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


#%% Neural Network Classifier

class NN(nn.Module):
    """
    Neural network for classifying.
    """
    def __init__(self,input_size,num_classes,middle_layers):
        super().__init__()
        self.fc1 = nn.Linear(input_size,middle_layers[0])
        self.fc2 = nn.Linear(middle_layers[0],middle_layers[1])
        #self.fc3 = nn.Linear(middle_layers[1],middle_layers[2])
        self.output = nn.Linear(middle_layers[1],num_classes)
 
    def forward(self, x):
        x = F.log_softmax(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.output(x)
        return F.softmax(x,dim=1)

class MyDataset(Dataset):
    """
    Creates the object as required by PyTorch DataLoader.
    Inputs:
        - feature
        - labels
    """
    def __init__(self, features, labels):
        self.labels = labels
        self.features = features
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        feature = self.features[idx]
        sample = {"X": feature, "y": label}
        return sample

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Hyperparameters
input_size = 79
batch_size = 1000
num_classes = 10
middle_layers = (128,64)
learning_rate = 1e-3
epochs = 10


### Load Data

size_class = 30000
merged = datasets[0].iloc[0:1]
for i in range(9):
    data = datasets[i]
    print("i=",i,"len(data)",len(data))
    if len(data)<size_class:
        file = data.iloc[0:1]
        while len(file)<size_class:
            file = file.append(data)
        merged= merged.append(file)
        print(len(file))
    else:
        file = data.iloc[0:size_class]
        merged = merged.append(file)
        print(len(file))

print("Size of training dataset:",len(merged))

# Randomly shuffle the data since now the rows in "merged" are ordered 0..0,1..1,2...
merged = merged.sample(frac=1)
X = merged[keys[:-1]]
y = merged[keys[-1]]

#Scale
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

# Change into tensors for pyTorch to be able to read it
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)


# DataLoader and MyDataset initialisations
train = MyDataset(X_train,y_train)
test = MyDataset(X_test,y_test)


trainset = DataLoader(train,batch_size=batch_size,shuffle=True)
testset = DataLoader(test,batch_size=batch_size,shuffle=True)


### Initialise network
model = NN(input_size,num_classes,middle_layers).to(device)


### Optimiser and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# =============================================================================
# if torch.cuda.is_available():
#         model.cuda()
#         criterion = criterion.cuda()
# =============================================================================


start  = time.time()

i=1
for epoch in range(epochs):
    for data in trainset:
        # data is a batch of featuresets and labels
        X,y = data["X"],data["y"]
        X = X.to(device=device)
        y = y.to(device=device)
        model.zero_grad()
        output = model(X.view(-1,input_size))
        #loss = F.nll_loss(output,y)
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()
        
    if i % int(epochs/10) == 0:
        print(f'Epoch: {i}/{epochs} Loss: {loss}')
        
    i += 1
 
# =============================================================================
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# 
# =============================================================================
end = time.time()
print("Time:", end-start)

### Accuracy/Result Analysis
def get_accuracy(dataset):
    """
    Gets overall accuracy of the given dataset
    which must be of type DataLoader.
    Returns accuracy to 3 s.f
    """
    
    correct=0
    total=0
    with torch.no_grad():
        for data in trainset:
            X,y = data["X"],data["y"]
            X = X.to(device=device)
            y = y.to(device=device)
            output = model(X.view(-1,input_size))
            for idx,array in enumerate(output):
                if torch.argmax(array) == y[idx]:
                    correct +=1
                total +=1
    acc = round(correct/total,3)
    return acc

acc = get_accuracy(trainset)

pred_labels = []
real_labels = []
with torch.no_grad():
    for data in testset:
        X,y = data["X"],data["y"]
        X = X.to(device=device)
        y = y.to(device=device)
        output = model(X.view(-1,input_size))
        for idx,array in enumerate(output):
            pred_label = torch.argmax(array).item()
            real_label = y[idx].item()
            pred_labels.append(pred_label)
            real_labels.append(real_label)


df = pd.DataFrame({'Labels': real_labels, 'Predictions': pred_labels})
df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Labels'], df['Predictions'])]

fals_pos = df[df["Labels"]==9]
fals_pos = fals_pos[fals_pos["Predictions"]==0]


accuracy = df['Correct'].sum() / len(df)
signals = df[df["Labels"]==0]


confusionmatrix = confusion_matrix(real_labels, pred_labels)
print(" Confusion matrix:\n",confusionmatrix)
sig_acc= signals['Correct'].sum() / (signals['Correct'].sum()+ len(fals_pos))

print("Overall Accuracy:",accuracy)
print("Classified signals accuracy:",sig_acc)
print("Signal detection accuracy:", signals['Correct'].sum() / len(signals))




