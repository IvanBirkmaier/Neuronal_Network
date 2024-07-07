from cgi import test
from copyreg import pickle
from fileinput import filename
from itertools import count
from operator import length_hint
from re import A
from sympy import O
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os
####################################################################################################################################################################################
######################################## Liest die erstellten Featuredaten als Datensatz ein. ######################################################################################
####################################################################################################################################################################################
df = pd.read_csv("data\features.csv")
originalDataframe =  df.copy()
####################################################################################################################################################################################
######################################## changeDataFrame. Erstellt aus der Kopie des eingelenen Dataframes   ######################################################################################
######################################## einen modifizierten der später genutzt wird um per Zufallsprinzip   ######################################################################################
######################################## Features herauszu selektieren um mit diesen ein Model zu Trainieren #####################################################################################
####################################################################################################################################################################################
def changeDataframe(dataf):
    cangeDataf = dataf.copy()
    cangeDataf["Numbre of all negativ Tweets"] = cangeDataf["Numbre of all negativ Tweets"].astype(np.float64)
    cangeDataf["Volume"] = cangeDataf["Volume"].astype(np.float64)
    cangeDataf["Numbre of all positiv Tweets"] = cangeDataf["Numbre of all positiv Tweets"].astype(np.float64)
    cangeDataf["Max. tweets in 1 Minute"] = cangeDataf["Max. tweets in 1 Minute"].astype(np.float64)
    cangeDataf["Min. tweets in 1 Minute"] = cangeDataf["Min. tweets in 1 Minute"].astype(np.float64)
    cangeDataf["Total number of tweets in intervall"] = cangeDataf["Total number of tweets in intervall"].astype(np.float64)
    cangeDataf.drop("Adj Close", axis=1, inplace=True)
    cangeDataf.drop("Interval", axis=1, inplace=True)
    cangeDataf.drop("Index", axis=1, inplace=True)
    cangeDataf.drop("Datum", axis=1, inplace=True)
    cangeDataf.drop("Close", axis=1, inplace=True)
    return cangeDataf
####################################################################################################################################################################################
######################################## generateRandomNumbre. Gibt eine Zufallszahl zurück die von einer startNumbre      ######################################################################################
######################################## bis zum Max. der möglichen features geht, um eine Anzahl an Features zu bestimmen ######################################################################################
######################################## die Später als Inputs verwendet werden können.                                     #####################################################################################
####################################################################################################################################################################################
def generateRandomNumbre(startNumbre, dataf):
    randomNumber = random.randrange(startNumbre, dataf.shape[1])
    if randomNumber <= dataf.shape[1]:
        return randomNumber
    else:
        newRandomNumber = generateRandomNumbre(dataf)
        return newRandomNumber
####################################################################################################################################################################################
######################################## decideHowManyFeatures. erstellt ein Dataframe von zufällig vielen Features     ######################################################################################
####################################################################################################################################################################################
def decideHowManyFeatures(dataf, dfForCloseandDate):
    howManyFeautures = generateRandomNumbre(3, dataf) #minimum was ich an Features haben will für ein Model
    dfWitchCoumns = []
    count = 0
    for i in range(howManyFeautures):
        if count >= 1:
            t = randomFeatures(dataf,dfWitchCoumns)
            dfWitchCoumns.append(t)
        else:
            count = count+1
            pickRandomColums = generateRandomNumbre(1,dataf)
            dfWitchCoumns.append(pickRandomColums)
    dataframeForTensor = []
    for i in dfWitchCoumns:
        dataframeForTensor.append(dataf.iloc[:,i])
    datafForTensorPrepared = pd.DataFrame(dataframeForTensor)
    datafForTensorPrepared = datafForTensorPrepared.swapaxes("index", "columns")
    close = dfForCloseandDate["Close"]
    date = dfForCloseandDate["Datum"]
    datafForTensorPrepared["Close"] = close
    datafForTensorPrepared["Datum"] = date
    return datafForTensorPrepared
###################################################################################################################################################################################
########################################  randomFeatures. Überwacht dass bei der estellung des in der oberen Methode  #############################################################
########################################  erstellten Dataframe keine Features zwei Mal Auftauchen.                    #############################################################
###################################################################################################################################################################################
def randomFeatures(dataf,dfWitchCoumns):
    pickRandomColums = generateRandomNumbre(1,dataf)
    if not (pickRandomColums in dfWitchCoumns):
        return  pickRandomColums
    else:
        return randomFeatures(dataf,dfWitchCoumns)
###################################################################################################################################################################################
########################################  datafForTensor ist eben genau dieser Datframe der nun weiter verarbeitet wird  ##########################################################
###################################################################################################################################################################################          
datafForTensor = decideHowManyFeatures(changeDataframe(originalDataframe),originalDataframe)
###################################################################################################################################################################################
########################################  Diese Schleifenlogig nimmt den Dataframe, gruppiert ihn nach Tagen in gleich   ##########################################################
########################################  große Teile und erstellt aus diesen Teilen ein numpy Array der später zu einem ##########################################################
########################################  benötigten Tensor ubesetzt wird.                                               ##########################################################
###################################################################################################################################################################################          
tensorsList = []
listOfLabels = []
for k, g in datafForTensor.groupby('Datum'):
    dataForCheck =  g["Close"].to_numpy()
    listOfLabels.append(dataForCheck)
    g.drop("Datum", axis=1, inplace=True)
    g.drop("Close", axis=1, inplace=True)
    inputData = g.to_numpy()
    tensorsList.append(inputData)
###################################################################################################################################################################################          
###################################### Unterteilung des Daten Satzes in die Splits: Train, Validation (jeweils 50% von 80% des Datensatzes), ######################################
###################################### Test (restlichen 20% des Datensatzes).                                                                ######################################
###################################### Aufteilung läuft nach dem Zufallsprinzip um Variation der Datensätze zu schaffen um verschiedenste    ######################################
###################################### Modele erzeugen zu können.                                                                            ######################################
###################################################################################################################################################################################          
length_totalDataset = len(tensorsList)
calculate_eightyPercentTrainVali = length_totalDataset * 0.8
calculate_fiftyPercentForTrainValiSplit = calculate_eightyPercentTrainVali / 2
length_trainAndValiDataset = round(calculate_fiftyPercentForTrainValiSplit)
train_data = []
train_label = []
vali_data = []
vali_label = []
while len(train_data) <= length_trainAndValiDataset-1:
    pick_data = random.randrange(0, len(tensorsList))
    train_data.append(tensorsList[pick_data])
    train_label.append(listOfLabels[pick_data])
    tensorsList.pop(pick_data)
    listOfLabels.pop(pick_data)
while len(vali_data) <= length_trainAndValiDataset-1:
    pick_data = random.randrange(0, len(tensorsList))
    vali_data.append(tensorsList[pick_data])
    vali_label.append(listOfLabels[pick_data])
    tensorsList.pop(pick_data)
    listOfLabels.pop(pick_data)
test_data = tensorsList
test_label = listOfLabels
###################################################################################################################################################################################          
###################################### Definition der Hypervariablen dynamisch die später im  Model benutzt werden. Auch hier passiert       ######################################
###################################### Dies bei einem großteil der Variablen nach einem Zufallsprinzip, um möglicht viel Variazion           ######################################
###################################### in den verschiedenen Modelen schaffen zu können.                                                      ######################################
###################################################################################################################################################################################          

# batch_size = 13
leraning_rate = 0.01

input_size = len(test_data[0][1])
print("input_size ",input_size)

sequence_length = len(test_data[0])
print("Squencesize ",sequence_length)

num_classes = sequence_length


def numOfLayer():
    num =  random.randrange(1,6)
    return num
num_layers = numOfLayer()
print("num_layerOne ", num_layers)

def numOfEpochs():
    num =  random.randrange(10,50)
    return num
num_epochs = numOfEpochs()
print("epochs ",num_epochs)

def numOfHiddenSize():
    num =  random.randrange(50,500)
    return num
hidden_size = numOfHiddenSize()
print("Hiddensize ",hidden_size)

def numOfDropoutRate():
    dropoutRates = [0.75,0.5,0.25]
    num = random.choice(dropoutRates)
    return num
dropout = numOfDropoutRate()
print("dropout ",dropout)

input_sizeForLayerTwo = hidden_size*2
print("input Layer 2 ",input_sizeForLayerTwo)




##################################################################################################################################################################
###################################             Inizialisierung des LSTM Modells                ###################################################################
###################################################################################################################################################################
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, input_sizeForLayerTwo):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.input_sizeForLayerTwo = input_sizeForLayerTwo
        self.layer_1 = nn.LSTM(input_size,hidden_size, num_layers, batch_first=True, dropout = 1,
                                bidirectional  = True) # input_size, hidden_size
        self.dropout = nn.Dropout(dropout)
        self.layer_L2 = nn.LSTM(input_sizeForLayerTwo ,hidden_size, num_layers, batch_first=True, dropout = 1,
                                bidirectional  = True) # input_size, hidden_size
        self.dropout2 = nn.Dropout(dropout)
        self.layer_2 = nn.Linear(hidden_size*2, 1) # input_size, output_size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers*2, x.size(0),self.hidden_size)
        #out, (h_n, c_n) = self.layer_1(x ,h0)
        out, (h_n, c_n) = self.layer_1(x, (h0, c0))
        dropout = self.dropout(out)
        out, _ = self.layer_L2(dropout,(h_n, c_n))
        dropout2 = self.dropout(out)
        output = torch.relu(self.layer_2(dropout2))
        changedia = torch.squeeze(output)
        return changedia

net = Net(input_size, hidden_size, num_layers, num_classes, input_sizeForLayerTwo)
net = net.float()

###############################################################################################################################################################################################################
############################################ Inizialisierung der Input Tensoren, des LOSS und des Optimizers. Zusätzlich der Erly stopping Methode,                 
########################################### die dafür genutzt werden soll um das Trainig zu beenden, wenn es keine Änderung im Loss der einzelnen Epochen gibt
###########################################  sprich das Model noch Mal angepasst werden muss. loss_vals speichert die Loswerte einer Epoche als Liste ab
###############################################################################################################################################################################
inputs = torch.tensor(train_data)
labels = torch.tensor(train_label)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=leraning_rate)

####################################################################################################################################################
############################################### Stopt das Training wenn sich mit beim Trainieren der Loss nicht mehr verändert #####################
#######################################################################################################################################################
def early_stopping(twoBeforeCurrentLoss, oneBeforeCurrentLoss, current_loss):
    if current_loss == oneBeforeCurrentLoss and current_loss == twoBeforeCurrentLoss:
            return True

################## Speichert für einen Trainigsdurchlauf alle Loss werte zum wiederverwende #######################################################################
loss_vals = list()


################################## Speiechert das Model einer jeden epoche################################################
def saveModel(model, filename):
    print("Model wurde gespeichert")
    torch.save(model, filename)


############################################################ Epochen durchlauf ############################################################
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs.float())
    loss = criterion(outputs.float(), labels.float())
    if epoch > 1 :
        checkIfearlyStopping = early_stopping(loss_vals[epoch -2], loss_vals[epoch -1 ], loss.item())
        if checkIfearlyStopping ==True:
            eralyepo = epoch-2
            print("Training abgebrochen. Da sich die Loss Werte von Epoche ",eralyepo," bis Epoche ", epoch, " nicht verändert haben. "+
            "Bitte Modelanpassungen vornehmen.")
            break
    loss.backward()
    optimizer.step()
    itemToString = str(loss.item())
    path = "results/models_with_loss_"+itemToString+".path.tar"
    saveModel(net, path)
    print("Epoche ",epoch+1," durchgelaufen! Mit einem Loss von: ", loss.item())
    loss_vals.append(loss.item())

def bestModelfromTraining():
    minLoss = min(loss_vals)
    return minLoss

def deleteAllModelsWhichArentTheBest():
    for i in loss_vals:
        if i != bestModelfromTraining():
            itemToString = str(i)
            path = "results/models_with_loss_"+itemToString+".path.tar"
            os.remove(path)
          
deleteModels = deleteAllModelsWhichArentTheBest()


def loadBestModel():
    itemToString = str(bestModelfromTraining())
    path = "results/models_with_loss_"+itemToString+".path.tar"
    net = torch.load(path)
    net.eval()
    return net 

netForVali = loadBestModel()

def modelValidation():
    return np.NaN

def modelTesting():
    return np.NaN
