import sys
import glob
import torch
from torch.utils import data
from utils import onsetCNN, Dataset
import numpy as np
import matplotlib.pyplot as plt

#function to repeat positive samples to improve data balance
def balance_data(ids, labels):
	ids2add=[]
	for idi in ids:
		if labels[idi]==1:
			ids2add.append(idi)
			ids2add.append(idi)
			ids2add.append(idi)
	return ids2add

#use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1")#torch.device("cuda:0" if use_cuda else "cpu")

#parameters for data loader
params = {'batch_size': 256,'shuffle': True,'num_workers': 6}
max_epochs = 50

#data
datadir='/media/Sharedata/rohit/SS_onset_detection/data_pt/'
songlist=np.loadtxt('songlist.txt',dtype=str)
labels = np.load('labels_master.npy').item()
weights = np.load('weights_master.npy').item()

#model
model=onsetCNN().double().to(device)
criterion=torch.nn.BCELoss(reduction='none')
optimizer=torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.45)
#optimizer=torch.optim.Adam(model.parameters(), lr=0.05)

#cross-validation loop
fold = int(sys.argv[1]) #cmd line argument
partition = {'all':[], 'train':[], 'validation':[]}
val_split = np.loadtxt('splits/8-fold_cv_random_%d.fold'%fold,dtype='str')
for song in songlist:
	ids = glob.glob(datadir+song+'/*.pt')
	if song in val_split:	partition['validation'].extend(ids)
	else:	partition['train'].extend(ids)

#balance data
#partition['train'].extend(balance_data(partition['train'],labels))

#print data balance percentage
n_ones=0.
for idi in partition['train']:
	if labels[idi]==1.: n_ones+=1
print('Fraction of positive examples: %f'%(n_ones/len(partition['train'])))

#generators
training_set = Dataset(partition['train'], labels, weights)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, weights)
validation_generator = data.DataLoader(validation_set, **params)

#training epochs loop
train_loss_epoch=[]
val_loss_epoch=[]
for epoch in range(max_epochs):
	train_loss_epoch+=[0]
	val_loss_epoch+=[0]

	##training
	n_train=0
	for local_batch, local_labels, local_weights in training_generator:
		n_train+=local_batch.shape[0]
		
		#transfer to GPU
		local_batch, local_labels, local_weights = local_batch.to(device), local_labels.to(device), local_weights.to(device)

		#update weights
		optimizer.zero_grad()
		outs = model(local_batch).squeeze()
		loss = criterion(outs, local_labels)
		loss = torch.dot(loss,local_weights)
		loss /= local_batch.size()[0]
		loss.backward()
		optimizer.step()
		train_loss_epoch[-1]+=loss.item()
	train_loss_epoch[-1]/=n_train
	
	##validation
	n_val=0
	with torch.set_grad_enabled(False):
		for local_batch, local_labels, local_weights in validation_generator:
			n_val+=local_batch.shape[0]

			#transfer to GPU
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)
			
			#evaluate model
			outs = model(local_batch).squeeze()
			loss = criterion(outs, local_labels).mean()
			val_loss_epoch[-1]+=loss.item()
	val_loss_epoch[-1]/=n_val

	#print loss in current epoch
	print('Epoch no: %d/%d\tTrain loss: %f\tVal loss: %f'%(epoch, max_epochs, train_loss_epoch[-1], val_loss_epoch[-1]))
	
	#update LR and momentum (only if using SGD)
	for param_group in optimizer.param_groups:
		param_group['lr'] *= 0.995
		if 10<=epoch<=20: param_group['momentum'] += 0.045

#plot losses vs epoch
plt.plot(train_loss_epoch,label='train')
plt.plot(val_loss_epoch,label='val')
plt.legend()
plt.savefig('./plots/loss_curves_%d'%fold)
plt.clf()
torch.save(model.state_dict(), 'saved_model_%d.pt'%fold)
