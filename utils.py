import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

#model
class onsetCNN(nn.Module):
	def __init__(self):
		super(onsetCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, (3,7))
		self.pool1 = nn.MaxPool2d((3,1))
		self.conv2 = nn.Conv2d(10, 20, 3)
		self.pool2 = nn.MaxPool2d((3,1))
		self.fc1 = nn.Linear(20 * 7 * 8, 256)
		self.fc2 = nn.Linear(256,1)
		self.dout = nn.Dropout(p=0.5)
    	
	def forward(self,x):
		y=torch.tanh(self.conv1(x))
		y=self.pool1(y)
		y=torch.tanh(self.conv2(y))
		y=self.pool2(y)
		y=self.dout(y.view(-1,20*7*8))
		y=self.dout(torch.sigmoid(self.fc1(y)))
		y=torch.sigmoid(self.fc2(y))
		return y

#data-loader(https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, weights):
        'Initialization'
        self.labels = labels
        self.weights = weights
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        #X = torch.tensor(np.load(ID))
        X = torch.load(ID)
        y = self.labels[ID]#.replace('.npy','')]
        w = self.weights[ID]#.replace('.npy','')]

        return X, y, w

#peak-picking function
def peakPicker(data, peakThresh):
	peaks=np.array([],dtype='int')
	for ind in range(1,len(data)-1):
		if ((data[ind+1] < data[ind] > data[ind-1]) & (data[ind]>peakThresh)):
			peaks=np.append(peaks,ind)
	return peaks

#merge onsets if too close - retain only stronger one
def merge_onsets(onsets,strengths,mergeDur):
	onsetLocs=np.where(onsets==1)[0]
	ind=1
	while ind<len(onsetLocs):
		if onsetLocs[ind]-onsetLocs[ind-1] < mergeDur:
			if strengths[onsetLocs[ind]]<strengths[onsetLocs[ind-1]]:
				onsets[onsetLocs[ind]]=0
				onsetLocs=np.delete(onsetLocs,ind)
			else:
				onsets[onsetLocs[ind-1]]=0
				onsetLocs=np.delete(onsetLocs,ind-1)
		else: ind+=1
	return onsets

#evaluate the hits and misses of predicted peaks
def eval_output(_outLabels, _outProbs, _groundTruth, _tolerance, _mergeDur):
	_outLabels=merge_onsets(_outLabels,_outProbs,_mergeDur)
	peakLocsOut=np.where(_outLabels==1.0)[0]
	peakLocsGt=np.where(_groundTruth==1.0)[0]
	nPositives=len(peakLocsGt)
	nTP=0

	for i_peak in range(len(peakLocsOut)):
		while(len(peakLocsGt) != 0):
			if abs(peakLocsOut[i_peak] - peakLocsGt[0]) <= int(_tolerance/2):
				peakLocsGt=np.delete(peakLocsGt,0)
				nTP+=1
				break
			elif peakLocsOut[i_peak] < peakLocsGt[0]:
				break
			else:
				peakLocsGt=np.delete(peakLocsGt,0)
			  
	nFP=len(peakLocsOut)-nTP
	return nTP, nFP, nPositives
