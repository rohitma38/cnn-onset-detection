import numpy as np
import os
import librosa
import torch

#function to zero pad ends of spectrogram
def zeropad2d(x,n_frames):
	y=np.hstack((np.zeros([x.shape[0],n_frames]), x))
	y=np.hstack((y,np.zeros([x.shape[0],n_frames])))
	return y

#function to create N-frame overlapping chunks of the full audio spectrogram  
def makechunks(x,duration):
	y=np.zeros([x.shape[1],x.shape[0],duration])
	for i_frame in range(x.shape[1]-duration):
		y[i_frame]=x[:,i_frame:i_frame+duration]
	return y

#data dirs
audio_dir='/media/Sharedata/rohit/SS_onset_detection/audio'
onset_dir='/media/Sharedata/rohit/SS_onset_detection/onsets'
save_dir='/media/Sharedata/rohit/SS_onset_detection/data_pt_test'

#data stats for normalization
stats=np.load('means_stds.npy')
means=stats[0]
stds=stats[1]

#context parameters
contextlen=7 #+- frames
duration=2*contextlen+1

#main
songlist=np.loadtxt('songlist.txt',dtype=str)
audio_format='.flac'
labels_master={}
weights_master={}
filelist=[]
for item in songlist:
	print(item)
	#load audio and onsets
	x,fs=librosa.load(os.path.join(audio_dir,item+audio_format), sr=44100)
	if not os.path.exists(os.path.join(onset_dir,item+'.onsets')): continue
	onsets=np.loadtxt(os.path.join(onset_dir,item+'.onsets'))
	
	#get mel spectrogram
	melgram1=librosa.feature.melspectrogram(x,sr=fs,n_fft=1024, hop_length=441,n_mels=80, fmin=27.5, fmax=16000)
	melgram2=librosa.feature.melspectrogram(x,sr=fs,n_fft=2048, hop_length=441,n_mels=80, fmin=27.5, fmax=16000)
	melgram3=librosa.feature.melspectrogram(x,sr=fs,n_fft=4096, hop_length=441,n_mels=80, fmin=27.5, fmax=16000)
	
	#log scaling
	melgram1=10*np.log10(1e-10+melgram1)
	melgram2=10*np.log10(1e-10+melgram2)
	melgram3=10*np.log10(1e-10+melgram3)
	
	#normalize
	melgram1=(melgram1-np.atleast_2d(means[0]).T)/np.atleast_2d(stds[0]).T
	melgram2=(melgram2-np.atleast_2d(means[1]).T)/np.atleast_2d(stds[1]).T
	melgram3=(melgram3-np.atleast_2d(means[2]).T)/np.atleast_2d(stds[2]).T
	
	#zero pad ends
	melgram1=zeropad2d(melgram1,contextlen)
	melgram2=zeropad2d(melgram2,contextlen)
	melgram3=zeropad2d(melgram3,contextlen)
	
	#make chunks
	melgram1_chunks=makechunks(melgram1,duration)
	melgram2_chunks=makechunks(melgram2,duration)
	melgram3_chunks=makechunks(melgram3,duration)
	
	#generate song labels
	hop_dur=10e-3
	labels=np.zeros(melgram1_chunks.shape[0])
	weights=np.ones(melgram1_chunks.shape[0])
	idxs=np.array(np.round(onsets/hop_dur),dtype=int)
	labels[idxs]=1
	
	#target smearing
	labels[idxs-1]=1
	labels[idxs+1]=1
	weights[idxs-1]=0.25
	weights[idxs+1]=0.25

	labels_dict={}
	weights_dict={}
	
	#save
	savedir=os.path.join(save_dir,item[:-5])
	if not os.path.exists(savedir): os.makedirs(savedir)
	
	for i_chunk in range(melgram1_chunks.shape[0]):
		savepath=os.path.join(savedir,str(i_chunk)+'.pt')
		#np.save(savepath,np.array([melgram1_chunks[i_chunk],melgram2_chunks[i_chunk],melgram3_chunks[i_chunk]]))
		torch.save(torch.tensor(np.array([melgram1_chunks[i_chunk], melgram2_chunks[i_chunk], melgram3_chunks[i_chunk]])), savepath)
		filelist.append(savepath)
		labels_dict[savepath]=labels[i_chunk]
		weights_dict[savepath]=weights[i_chunk]

	#append labels to master
	labels_master.update(labels_dict)
	weights_master.update(weights_dict)
	
	np.savetxt(os.path.join(savedir,'labels.txt'),labels)
	np.savetxt(os.path.join(savedir,'weights.txt'),weights)
	
np.save('labels_master',labels_master)
np.save('weights_master',weights_master)
#np.savetxt('filelist.txt',filelist,fmt='%s')
