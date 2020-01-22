import sys
import glob
import torch
from torch.utils import data
from utils import onsetCNN, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

#Use gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

#evaluation tolerance and merge duration for close onsets
tolerance=60e-3 #+- tolerance/2 seconds
mergeDur=20e-3
hop_dur=10e-3
mergeDur_frame=mergeDur/hop_dur
tolerance_frame=tolerance/hop_dur

fold = int(sys.argv[1]) #cmd line argument

#load model
path_to_saved_model = 'models/saved_model_%d.pt'%fold
model = onsetCNN().double().to(device)
model.load_state_dict(torch.load(path_to_saved_model))
model.eval()

#data
datadir='/media/Sharedata/rohit/SS_onset_detection/data_pt/'
#songlist=os.listdir(datadir)
songlist=np.loadtxt('splits/8-fold_cv_random_%d.fold'%fold,dtype=str)
labels = np.load('labels_master_test.npy').item()

#loop over test songs
scores=np.array([])
n_songs=len(songlist)
i_song=0
for song in songlist:
	print('%d/%d songs\n'%(i_song,n_songs))
	i_song+=1
	
	odf=np.array([])
	gt=np.array([])

	#generate frame-wise labels serially for song
	n_files=len(glob.glob(os.path.join(datadir,song+'/*.pt')))
	for i_file in range(n_files):
		x=torch.load(os.path.join(datadir,song+'/%d.pt'%i_file)).to(device)
		x=x.unsqueeze(0)
		y=model(x).squeeze().cpu().detach().numpy()
		odf=np.append(odf,y)
		gt=np.append(gt,labels[os.path.join(datadir,song+'/%d.pt'%i_file)])

	#evaluate odf
	scores_thresh=np.array([])
	#loop over different peak-picking thresholds to optimize F-score
	for predict_thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
		odf_labels=np.zeros(len(odf))
		
		#pick peaks
		odf_labels[utils.peakPicker(odf,predict_thresh)]=1.
		
		#evaluate, get #hits and #misses
		scores_thresh=np.append(scores_thresh,utils.eval_output(odf_labels, odf, gt, tolerance_frame, mergeDur_frame))
	
	#accumulate hits and misses for every song	
	if len(scores)==0: scores=np.atleast_2d(np.array(scores_thresh))
	else: scores=np.vstack((scores,np.atleast_2d(np.array(scores_thresh))))

#add hits and misses over all songs (to compute testset P, R and F-score)
scores=np.sum(scores,0)

# Write to file
fout=open('hr_fa_folds.txt','a')
for item in scores:
	fout.write('%d\t'%item)
fout.write('\n')
fout.close()
