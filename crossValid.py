import json
from itertools import product
import os
import numpy as np

import train
import load


file = open("myJson.json","r")
file2 = open("bestHyperParam.json","w")
param = json.load(file)

if not os.path.isfile('scaledTrainData.csv'):
	print("Scaling DATA ....")
	os.system('python scale.py')

X,Y = load.load_data(5000,'scaledTrainData.csv')
X_cv,Y_cv = load.load_data(100,'scaledCVData.csv')

best = -np.inf
bestHyperParam = ""

for LR,H_SIZE,EPOCHS,LAYERS in product(param["LR"],param["H_SIZE"],param["EPOCHS"],param["LAYERS"]):
	
	hyperParam = json.dumps({"LR":LR,"H_SIZE":H_SIZE,"EPOCHS":EPOCHS,"LAYERS":LAYERS})
	loss = train.train(X,Y,hyperParam,X_cv,Y_cv)
	print(hyperParam)
	print(loss)
	if best < loss:
		best = loss
		bestHyperParam = hyperParam

print("############# Finished ################")
print(bestHyperParam)
file2.write(bestHyperParam)

file.close()
file2.close()
