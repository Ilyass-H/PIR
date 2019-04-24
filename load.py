import sys
import json
import pandas as pd
import numpy as np
import torch
from torch import nn

IN_SIZE = 8							# number of features

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data(N,file,flag="valid"):

	d = pd.read_csv(file)
	dd = None
	if flag == "test":
		dd = pd.read_csv("DH8D_test.csv.xz")[['segment','massPast']]

	#first segment id
	beg = d['segment'][0].item()
	# Find the lengths (number of points) of each trajectory in the batch  
	# And Sort in decreasing order to use in pack_padded_sequence
	lengths_id = sorted(list(range(beg,beg+N)), key = lambda i : d[d.segment == i].count()[0],reverse=True)
	lengths = list(d[d.segment == i].count()[0] for i in lengths_id if d[d.segment == i].count()[0] > 0)
	N = len(lengths)
	lengths_id = lengths_id[0:N]
	MAX_POINTS = lengths[0]
	X = np.zeros((N,MAX_POINTS,IN_SIZE))
	Y = np.zeros((N,MAX_POINTS,1))
	massPast = np.zeros((N,MAX_POINTS,1))

	for i in range(N):	
		ind = lengths_id[i]
		length = lengths[i]

		X[i][:length] = d[d.segment == ind].drop(['segment','massFutur'],axis=1).values
		
		m = (d[d.segment == ind]['massFutur']).values
		if flag == "test":
			mp = (dd[dd.segment == ind]['massPast']).values
			mp.shape = [len(mp),1]
			massPast[i][:length] = mp[:length]
		m.shape = [len(m),1]
		Y[i][:length] = m
		

	X = nn.utils.rnn.pack_padded_sequence(torch.from_numpy(X).type(torch.float), lengths, batch_first=True).to(device)
	Y = torch.from_numpy(Y).type(torch.float).to(device)
	if flag == "test":
		return X,Y,massPast
	else:
		return X,Y
