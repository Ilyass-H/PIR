import pandas as pd
import numpy as np
import json

import train
import load

MEAN = 26331.311502
STD = 1799.268739

N_TEST = 10#10279
N_Train = 10#21896

log = open("log", "a")

print("\n\n######### Start Testing ############# \n")
log.write("\n\n######### Start Testing ############# \n\n")

hp = open("bestHyperParam.json","r")
hyperParam = json.load(hp)

print("Loading Test Data ...\n")
log.write("Loading Test Data ...\n\n")
X_test,Y_test,massPast = load.load_data(N_TEST,'scaledTestData.csv',"test")

print("Loading Training Data ...\n")
log.write("Loading Training Data ...\n\n")
X,Y = load.load_data(N_Train,'scaledTrainTestData.csv')

hyperParam = json.dumps({"LR":hyperParam["LR"],"H_SIZE":hyperParam["H_SIZE"],"EPOCHS":hyperParam["EPOCHS"],"LAYERS":hyperParam["LAYERS"]})

print("Start Training ... \n")
pred = train.train(X,Y,hyperParam,X_test,Y_test,"test")


pred_loss = 0
bada_loss = 0

cnt = 4

print("\n\n")
log.write("\n\n\n")
log.write("prediction,massFutur,massPast")
log.write("\n")
for j in range(len(pred)):
	for i in range(10,len(pred[0])):
		if(Y_test[j][i] != 0):
			if cnt > 0:
				p = int((pred[j][i].item() * STD)+MEAN)
				y = int(((Y_test[j][i].item() * STD)+MEAN))
				m = int(massPast[j][i][0])
				print(p,end=' - ')
				log.write(str(p))
				log.write(" - ")
				print(y,end=",  massPast = ")
				log.write(str(y))
				log.write(",  massPast = ")
				print(massPast[j][i][0])
				log.write(str(m))
				log.write("\n")
				
			pred_loss += (((pred[j][i].item() * STD)+MEAN) - ((Y_test[j][i][0] * STD)+MEAN))**2
			bada_loss += (massPast[j][i][0] - ((Y_test[j][i][0] * STD)+MEAN))**2
	if cnt > 0:
		print("\n######################\n")
		log.write("\n######################\n\n")
		cnt-=1


print("\n########## RMSE LOSS : \n")
log.write("\n########## RMSE LOSS : \n\n")
print("\n# pred_loss = ",np.sqrt(pred_loss.cpu()))
log.write("\n# pred_loss = ")
log.write(str(np.sqrt(pred_loss.cpu().item())))
print("# bada_loss = ",np.sqrt(bada_loss.cpu()))
log.write("\n# bada_loss = ")
log.write(str(np.sqrt(bada_loss.cpu().item())))