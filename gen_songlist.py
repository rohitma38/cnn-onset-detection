import numpy as np
import os

songlist=np.array([])
for foldfile in os.listdir('./splits/'):
	foldsongs=np.loadtxt('./splits/'+foldfile,dtype=str)
	songlist=np.append(songlist,foldsongs)

np.savetxt('songlist.txt',songlist,fmt='%s')
