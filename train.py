import sys
import json
import pandas as pd
import numpy as np
import torch
from torch import nn

IN_SIZE = 8


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.get_device_name(device))

def train(X,Y,HyperParam,X_v,Y_v):


	if HyperParam == None :
		HyperParam = {"LR": 0.001, "H_SIZE": 10, "EPOCHS": 5000, "LAYERS": 1}
	else:
		HyperParam = json.loads(HyperParam)


	class LSTM(nn.Module):
		def __init__(self):
			super(LSTM,self).__init__()

			self.lstm = nn.LSTM(
				input_size = IN_SIZE,
				hidden_size = HyperParam["H_SIZE"],
				num_layers = HyperParam["LAYERS"],
				batch_first = True)

			self.linear = nn.Linear(HyperParam["H_SIZE"], 1)

		def init_hidden(self,n):
			return (torch.zeros(HyperParam["LAYERS"], n, HyperParam["H_SIZE"]),
	        		torch.zeros(HyperParam["LAYERS"], n, HyperParam["H_SIZE"]))

		def forward(self,x):
	   		out,self.h = self.lstm(x)
	   		out = np.array(nn.utils.rnn.pad_packed_sequence(out, batch_first=True))
	   		y = self.linear(out[0])
	   		return y

	myLSTM = LSTM().to(device)

	optimizer = torch.optim.Adam(myLSTM.parameters(), lr=HyperParam["LR"])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	loss_func = nn.MSELoss().to(device)


	# Fitting du modele
	for step in range(HyperParam["EPOCHS"]):

		myLSTM.h = myLSTM.init_hidden(len(X))
		prediction = myLSTM(X)
		loss = loss_func(prediction, Y)
		
		#if step % 500 == 0:
		#    print("step: ", step, "MSE: ", loss.item())

		optimizer.zero_grad()                   
		loss.backward()
		optimizer.step()
		scheduler.step(loss.item())
	

	# Testing on Cross-Validation Data Set
	myLSTM.h = myLSTM.init_hidden(len(X_v))
	prediction = myLSTM(X_v)
	return loss_func(prediction,Y_v)
