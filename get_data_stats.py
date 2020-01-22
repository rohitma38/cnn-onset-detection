import numpy as np
import os
import librosa

#data dir
audio_dir='/media/Sharedata/rohit/SS_onset_detection/audio'

#main
songlist=np.loadtxt('songlist.txt',dtype=str)
i_song=0

means_song=[np.array([]),np.array([]),np.array([])]
stds_song=[np.array([]),np.array([]),np.array([])]

for i_song in range(len(songlist)):
	#load audio
	x,fs=librosa.load(os.path.join(audio_dir,songlist[i_song]+'.flac'), sr=44100)
	
	#get mel spectrogram
	melgram1=librosa.feature.melspectrogram(x,sr=fs,n_fft=1024, hop_length=441,n_mels=80, fmin=27.5, fmax=16000)
	melgram2=librosa.feature.melspectrogram(x,sr=fs,n_fft=2048, hop_length=441,n_mels=80, fmin=27.5, fmax=16000)
	melgram3=librosa.feature.melspectrogram(x,sr=fs,n_fft=4096, hop_length=441,n_mels=80, fmin=27.5, fmax=16000)
	
	#log scaling
	melgram1=10*np.log10(1e-10+melgram1)
	melgram2=10*np.log10(1e-10+melgram2)
	melgram3=10*np.log10(1e-10+melgram3)
	
	#compute mean and std of dataset
	if i_song==0:
		means_song[0]=np.mean(melgram1,1)
		means_song[1]=np.mean(melgram2,1)
		means_song[2]=np.mean(melgram3,1)

		stds_song[0]=np.std(melgram1,1)
		stds_song[1]=np.std(melgram2,1)
		stds_song[2]=np.std(melgram3,1)

	else:
		means_song[0]+=np.mean(melgram1,1)
		means_song[1]+=np.mean(melgram2,1)
		means_song[2]+=np.mean(melgram3,1)

		stds_song[0]+=np.std(melgram1,1)
		stds_song[1]+=np.std(melgram2,1)
		stds_song[2]+=np.std(melgram3,1)

means_song[0]/=i_song
means_song[1]/=i_song
means_song[2]/=i_song

stds_song[0]/=i_song
stds_song[1]/=i_song
stds_song[2]/=i_song

np.save('means_stds', np.array([means_song,stds_song]))
