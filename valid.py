import json
from itertools import product
import os
import numpy as np

import train
import load
import sys

N_Train = 11#000
N_valid = 4#000

file = open("hyperParameters.json","r")
file2 = open("bestHyperParam.json","w")
log = open("log", "w")
param = json.load(file)

if not os.path.isfile('scaledTrainData.csv'):
	print("Scaling DATA ....")
	log.write("Scaling DATA ....")
	os.system('python scale.py')


print("Loading Data for Validation ... ")
log.write("Loading Data for Validation ... ")

X,Y = load.load_data(N_Train,'scaledTrainData.csv')
X_v,Y_v = load.load_data(N_valid,'scaledValidData.csv')

print("Data Loaded \n\n")
log.write("Data Loaded \n\n")

best = np.inf
bestHyperParam = ""

print("############## Start Validation ################### \n\n")
log.write("############## Start Validation ################### \n\n")
for LR,H_SIZE,EPOCHS,LAYERS in product(param["LR"],param["H_SIZE"],param["EPOCHS"],param["LAYERS"]):
	
	hyperParam = json.dumps({"LR":LR,"H_SIZE":H_SIZE,"EPOCHS":EPOCHS,"LAYERS":LAYERS})
	loss = train.train(X,Y,hyperParam,X_v,Y_v,"valid")
	print("hyper paramerters : " + hyperParam)
	log.write("hyper paramerters : " + hyperParam)
	print("Loss : ",loss.item())
	log.write("Loss : ")
	log.write(str(loss.item()))
	print("\n")
	if loss < best:
		best = loss
		bestHyperParam = hyperParam

print("\n############# Finished Validation ################\n")
log.write("\n############# Finished Validation ################\n")
print("best Hyper Parameters : " + bestHyperParam)
log.write("best Hyper Parameters : " + bestHyperParam)
file2.write(bestHyperParam)

file.close()
file2.close()
