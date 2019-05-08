import os
import copy as cp
from pylab import *
import pretty_midi
import librosa             # The librosa library
import librosa.display     # librosa's display module (for plotting features)
import IPython.display     # IPython's display module (for in-line audio)
import matplotlib.pyplot as plt # matplotlib plotting functions
import matplotlib.style as ms   # plotting style
import numpy as np              # numpy numerical functions
import scipy
import math
import random
import csv
import copy

class Guichu_Generator():

	def __init__(self,music,video,time_coef=100,beat_coef=0.1,motion_coef=0.1,jump_threshold=5,jump_rate=1.0,path_select=0.3,block_size=32):
		super().__init__()
		self.init_gen(music,video,time_coef=time_coef,beat_coef=beat_coef,motion_coef=motion_coef,jump_threshold=jump_threshold,jump_rate=jump_rate,path_select=path_select,block_size=block_size)

	def init_gen(self,music,video,time_coef=100,beat_coef=0.1,motion_coef=0.1,jump_threshold=5,jump_rate=1.0,path_select=0.3,block_size=32):
		self.music = music
		self.video = video
		self.time_coef = time_coef
		self.beat_coef = beat_coef
		self.motion_coef = motion_coef
		self.jump_threshold = jump_threshold
		self.jump_rate = jump_rate
		self.path_select = path_select
		self.block_size = block_size
		self.matrix = None
		self.img = None


	#distance function for dynamic programming, x is for audio, y is for video
	#x,y is 2d vector, with first entry represents time, second entry represents label
	def distance(self,x,y,bias):
		x0 = bias[1]
		y0 = bias[0]
		time_diff = (x[0]-y[0]+x0-y0)**2
		corr1 = x[1]*y[1]
		corr2 = x[2]*y[2]
		if corr1 > 0:
			beat_diff = -math.sqrt(corr1)
		else:
			beat_diff = math.sqrt(-corr1)
		if corr2 > 0:
			motion_diff = -math.sqrt(corr2)
		else:
			motion_diff = math.sqrt(-corr2)
		#print([time_diff*c0,left_d_diff*c1,right_d_diff*c2,motion_diff*c3])
		d = self.time_coef*time_diff + self.beat_coef*beat_diff + self.motion_coef*motion_diff
		if d>1000:
			return 1000
		return d

	#penalty for jump, signoid function
	def jump(self,end,start):
		d = abs(end-start)
		penalty = self.jump_rate*math.exp(d-self.jump_threshold)/(1+math.exp(d-self.jump_threshold))
		return penalty

	#cut the music into pieces
	#return one list of feature series(beat) for each piece of music
	def cut_music(feature_set,num_of_bar,bar_per_cut):
		beats_per_bar = math.ceil(len(feature_set)/num_of_bar)
		pieces = []
		if num_of_bar%bar_per_cut == 0:
			for i in range(int(num_of_bar/bar_per_cut)-1):
				pieces.append(feature_set[i*beats_per_bar*bar_per_cut:(i+1)*beats_per_bar*bar_per_cut])
			pieces.append(feature_set[(i+1)*beats_per_bar*bar_per_cut:])
		else:
			for i in range(int(num_of_bar/bar_per_cut)):
				pieces.apeend(feature_set[i*beats_per_bar*bar_per_cut:(i+1)*beats_per_bar*bar_per_cut])
			pieces.append(feature_set[(i+1)*beats_per_bar*bar_per_cut:])
		return pieces

	def local_DTW(self,music_feature,video_feature):
		min_list = []
		alignment_list = []
		c0 = 0.7 #relavent coefficient
		c1 = 0.3 #relavent coefficient
		q = [] #distance matrix
		path = [] #record the point need to be include for alignment, 0 means left,
		          #-1 means up, -2 means up-left, other number N means jump from Nth column
		starting = [] #store the starting point for optimal path
		music = music_feature #input beat array
		video = video_feature #substitute beat array

		#initial q, path
		for i in range(len(music)):
			q.append([])
			path.append([])
			starting.append([])
			for j in range(len(video)):
				q[i].append(0)
				path[i].append(0)
				starting[i].append([0,0])
	            
		#compute distance matrix
		for i in range(len(music)):
			q[i][0] = 5000000000
			path[i][0] = -1
			starting[i][0] = [music[i][0],video[0][0]]

		for j in range(len(video)):
			q[0][j] = 0
			path[0][j] = 0
			starting[0][j] = [music[0][0],video[j][0]]
	    
		for i in range(1,len(music)):
			for j in range(1,len(video)):
				temp1 = q[i-1][j] + self.distance(music[i],video[j],starting[i-1][j])
				temp2 = q[i][j-1] + self.distance(music[i],video[j],starting[i][j-1])
				temp3 = q[i-1][j-1] + self.distance(music[i],video[j],starting[i-1][j-1])
				if (temp1 > temp2):
					if (temp2 > temp3):
						q[i][j] = temp3
						path[i][j] = -2
						starting[i][j] = starting[i-1][j-1]
					else:
						q[i][j] = temp2
						path[i][j] = 0
						starting[i][j] = starting[i][j-1]
				else:
					if (temp1 > temp3):
						q[i][j] = temp3
						path[i][j] = -2                        
						starting[i][j] = starting[i-1][j-1]
					else:
						q[i][j] = temp1
						path[i][j] = -1
						starting[i][j] = starting[i-1][j]
		#print(distance(music[10],video[1000],starting[9][1000]))
		#print(distance(music[10],video[2000],starting[10][1999]))


		#find minimum distance within matrix
		#note we want the end of the sub BGM almost fit the end of ori BGM
		#tolerance = 40 #represnet the range we can tolerent for the end
		dis_list = []
		for i in range(len(video)):
			dis_list.append(q[len(music)-1][i])
		index = []
		for i in range(len(video)):
			index.append(i)
		tup = sorted((i,j) for i,j in zip(dis_list,index))
		#print(tup)
		#while (tup[i][1] < len(ori) - tolerance):
		#    i = i+1

	    
		result = []
		num_path = int(len(tup)*self.path_select)
		#recover num_path best path
		for k in range(num_path):
			alignment = [] #store the points for alignment
			minimum = tup[k][0]
			min_index = tup[k][1]
			i = len(music)-1
			j = min_index
			while (i!=0 and j!=0):
				if (path[i][j] == 0):
					j = j-1
					alignment.append([music[i][0],video[j+1][0]])
				elif (path[i][j] == -1):
					i = i-1
					alignment.append([music[i+1][0],video[j][0]])
				elif (path[i][j] == -2):
					i = i-1
					j = j-1
					alignment.append([music[i+1][0],video[j+1][0]])
				else:
					alignment.append([music[i][0],video[j][0]])
					j = path[i][j]
					i = i-1
			alignment.append([music[0][0],video[j][0]])
			min_list.append(minimum)
			result.append([alignment[-1][1],alignment[0][1],minimum,alignment])
		#p = np.array(q)
		#plt.matshow(p)
		return result, np.array(q)

	#to make the matplot more clear (get rid of the part with very large error)
	def filter(m,threshold):
		filter_m = []
		for i in range(len(m.T)):
			if sum(m.T[i])/len(m.T)<threshold:
				filter_m.append(m.T[i])
		filter_m = np.array(filter_m)
		return filter_m.T

	def DTW(self):
		min_list = []
		alignment_list = []
		q = []
		linked_path = [] #path matrix
		info = [] #each element contains certain portion of paths for the music fragment
		music = self.music #input beat array
		video = self.video #substitute beat array
		for i in range(int(math.ceil(float(len(music))/self.block_size))):
			temp_music = music[i*self.block_size:min([(i+1)*self.block_size,len(music)])]
			temp_info, temp_q = self.local_DTW(temp_music,video)
			info.append(temp_info)
			print(temp_info[0][2])
			if i == 0:
				path_matrix = temp_q
			else:
				path_matrix = np.concatenate((path_matrix,temp_q),axis=0)
	    
		#print(info)
		#the global DTW list
		row_size = len(info)
		column_size = len(info[0])
		print(row_size,column_size)
	    
		for i in range(row_size):
			q.append([])
			linked_path.append([])
			for j in range(column_size):
				q[i].append(0)
				linked_path[i].append(-1)
	    
		for j in range(column_size):
			q[0][j] = info[0][j][2]
	        
		for i in range(1,row_size):    
			for j in range(column_size):
				temp_min = q[i-1][0] + self.jump(info[i-1][0][1],info[i][j][0]) + info[i][j][2]
				temp_index = 0
				for k in range(column_size):
					if temp_min > (q[i-1][k] + self.jump(info[i-1][k][1],info[i][j][0]) + info[i][j][2]):
						temp_min = q[i-1][k] + self.jump(info[i-1][k][1],info[i][j][0]) + info[i][j][2]
						temp_index = k
				linked_path[i][j] = temp_index
				q[i][j] = temp_min

		for i in range(len(q)):
		#print(jump(info[0][0][1],info[1][5][0],rate=0.01))
			print(min(q[i]))
			print(max(q[i]))
	            
		dis_list = []
		for i in range(column_size):
			dis_list.append(q[row_size-1][i])
		index = []
		for i in range(column_size):
			index.append(i)
		tup = sorted((i,j) for i,j in zip(dis_list,index))
		#print(tup)
		i=0
		#while (tup[i][1] < len(ori) - tolerance):
			#i = i+1
		minimum = tup[i][0]
		min_index = tup[i][1]
	    
	    
		path = [] #the path for blocks
		time_pt = []
		#recover path
		i = row_size-1
		j = min_index
		for i in range(row_size):
			path.append(j)
			time_pt.append(info[row_size-i-1][path[i]][1])
			j = linked_path[row_size-i-1][j]

		print(path)
		print(time_pt)
		#print(info[-1][path[0]])
		alignment = info[row_size-1][path[0]][3]
		for i in range(row_size-1):
			alignment += info[row_size-i-2][path[i+1]][3]
		#print(path_matrix[4])
		#filter_path_matrix = filter(path_matrix,50)
		alignment_list.append(alignment)
		min_list.append(minimum)
		#print(filter_path_matrix.shape)
		self.matrix = path_matrix[:,self.block_size:]
		self.img = plt.matshow(path_matrix[:,self.block_size:])
		#plt.matshow(filter_path_matrix)
		#print(alignment_list)
		#print(len(alignment_list[0]))
		with open('alignment.txt', 'w') as f:
			for item in alignment_list:
				f.write("%s\n" % item)
		return alignment_list, path_matrix


