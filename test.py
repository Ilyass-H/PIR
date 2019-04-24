import pandas as pd
import numpy as np
import json

import train
import load

MEAN = 26331.311502
STD = 1799.268739

N_TEST = 10279
N_Train = 21896


file = open("bestHyperParam.json","r")
hyperParam = json.load(file)

print("Loading Test Data ...\n")
X_test,Y_test,massPast = load.load_data(N_TEST,'scaledTestData.csv',"test")
print("Loading Training Data ...\n")
X,Y = load.load_data(N_Train,'scaledTrainTestData.csv')

hyperParam = json.dumps({"LR":hyperParam["LR"],"H_SIZE":hyperParam["H_SIZE"],"EPOCHS":hyperParam["EPOCHS"],"LAYERS":hyperParam["LAYERS"]})

print("Start Training ... \n")
pred = train.train(X,Y,hyperParam,X_test,Y_test,"test")


pred_loss = 0
bada_loss = 0

cnt = 3

print("\n\n")
for j in range(len(pred)):
	for i in range(10,len(pred[0])):
		if(Y_test[j][i] != 0):
			if cnt > 0:
				print(((pred[j][i].item() * STD)+MEAN),end=' - ')
				print(((Y_test[j][i][0].item() * STD)+MEAN),end=", massPast = ")
				print(massPast[j][i][0])
				
			pred_loss += (((pred[j][i].item() * STD)+MEAN) - ((Y_test[j][i][0] * STD)+MEAN))**2
			bada_loss += (massPast[j][i][0] - ((Y_test[j][i][0] * STD)+MEAN))**2
	if cnt > 0:
		print("\n######\n")
		cnt-=1


print("\n# pred_loss = ",np.sqrt(pred_loss.cpu()))
print("# bada_loss = ",np.sqrt(bada_loss.cpu()))
